from __future__ import annotations

from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kinesis as kinesis
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_lambda_event_sources as event_sources
from aws_cdk import aws_logs as logs
from aws_cdk import aws_sqs as sqs
from constructs import Construct

REPO_ROOT = Path(__file__).resolve().parent.parent

# Package lambda_fn/ and shared/ together from the repo root so there is exactly
# one copy of the feature code. Everything else is excluded from the asset.
LAMBDA_ASSET_EXCLUDES = [
    "**",
    ".*",
    "**/.*",
    "!lambda_fn",
    "!lambda_fn/**",
    "!shared",
    "!shared/**",
    "**/__pycache__",
    "**/*.pyc",
]


class RealtimePipelineStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        stream = kinesis.Stream(
            self,
            "ClickstreamStream",
            shard_count=1,
            retention_period=Duration.hours(24),
            encryption=kinesis.StreamEncryption.MANAGED,
            stream_mode=kinesis.StreamMode.PROVISIONED,
        )
        stream.apply_removal_policy(RemovalPolicy.DESTROY)

        table = dynamodb.Table(
            self,
            "FeatureTable",
            partition_key=dynamodb.Attribute(name="user_id", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="window_min", type=dynamodb.AttributeType.NUMBER),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            time_to_live_attribute="expires_at",
            removal_policy=RemovalPolicy.DESTROY,
        )

        dlq = sqs.Queue(
            self,
            "PoisonPillDlq",
            retention_period=Duration.days(14),
            encryption=sqs.QueueEncryption.SQS_MANAGED,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
        )

        log_group = logs.LogGroup(
            self,
            "ConsumerLogs",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=RemovalPolicy.DESTROY,
        )

        consumer = lambda_.Function(
            self,
            "StreamConsumer",
            runtime=lambda_.Runtime.PYTHON_3_12,
            architecture=lambda_.Architecture.ARM_64,
            handler="lambda_fn.lambda_function.lambda_handler",
            code=lambda_.Code.from_asset(str(REPO_ROOT), exclude=LAMBDA_ASSET_EXCLUDES),
            memory_size=512,
            timeout=Duration.seconds(30),
            log_group=log_group,
            environment={"TABLE_NAME": table.table_name, "DLQ_URL": dlq.queue_url},
        )

        # Least privilege: only the exact actions on the exact resources used.
        # GetItem is only used on the idempotency conflict path.
        consumer.add_to_role_policy(
            iam.PolicyStatement(
                actions=["dynamodb:UpdateItem", "dynamodb:GetItem"],
                resources=[table.table_arn],
            )
        )
        consumer.add_to_role_policy(
            iam.PolicyStatement(actions=["sqs:SendMessage"], resources=[dlq.queue_arn])
        )

        # KinesisEventSource grants the read actions the event source mapping
        # needs on this stream only (GetRecords, GetShardIterator, DescribeStream,
        # ListShards). Exhausted retries are sent to the same DLQ for inspection.
        consumer.add_event_source(
            event_sources.KinesisEventSource(
                stream,
                starting_position=lambda_.StartingPosition.LATEST,
                batch_size=100,
                max_batching_window=Duration.seconds(1),
                report_batch_item_failures=True,
                bisect_batch_on_error=True,
                retry_attempts=5,
                max_record_age=Duration.hours(1),
                on_failure=event_sources.SqsDlq(dlq),
            )
        )

        cloudwatch.Alarm(
            self,
            "ConsumerErrorsAlarm",
            alarm_description="Stream consumer Lambda reported errors",
            metric=consumer.metric_errors(period=Duration.minutes(5), statistic="Sum"),
            threshold=1,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )

        cloudwatch.Alarm(
            self,
            "IteratorAgeAlarm",
            alarm_description="Consumer is falling behind the Kinesis stream (>= 60s)",
            metric=stream.metric_get_records_iterator_age_milliseconds(
                period=Duration.minutes(1), statistic="Maximum"
            ),
            threshold=60_000,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )

        cloudwatch.Alarm(
            self,
            "DlqDepthAlarm",
            alarm_description="Poison pills or exhausted retries are landing in the DLQ",
            metric=dlq.metric_approximate_number_of_messages_visible(
                period=Duration.minutes(5), statistic="Maximum"
            ),
            threshold=1,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )

        CfnOutput(self, "StreamName", value=stream.stream_name)
        CfnOutput(self, "TableName", value=table.table_name)
        CfnOutput(self, "DlqUrl", value=dlq.queue_url)
        CfnOutput(self, "ConsumerFunctionName", value=consumer.function_name)
