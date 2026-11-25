#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import aws_cdk as cdk

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline_stack import RealtimePipelineStack  # noqa: E402

app = cdk.App()
RealtimePipelineStack(
    app,
    "RealtimeFeaturePipeline",
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION"),
    ),
)
app.synth()
