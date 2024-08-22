# Telemetry

Distilabel uses telemetry to report anonymous usage and error information. As an open-source software, this type of information is important to improve and understand how the product is used.

## How to opt-out

Telemetry is based on the [Hugging Face Hub telemetry](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/utilities#huggingface_hub.utils.send_telemetry). This means you can opt-out of telemetry by setting the environment variable `HF_HUB_DISABLE_TELEMETRY=1` or `HF_HUB_OFFLINE=1`. This will disable all telemetry reporting.

## Why report telemetry

Anonymous telemetry information enables us to continuously improve the product and detect recurring problems to better serve all users. We collect aggregated information about general usage and errors. We do NOT collect any information on users' data datasets, or metadata information.

## Sensitive data

We do not collect any piece of information related to potentially private input or output data. We don't identify individual users. Your data does not leave your environment at any point.

## Information reported

The following usage and error information is reported:

- The version of the library
- The Python version, e.g. `3.12.1`
- The system/OS name, such as `Linux`, `Darwin`, `Windows`
- The system’s release version, e.g. `Darwin Kernel Version 21.5.0: Tue Apr 26 21:08:22 PDT 2022; root:xnu-8020`
- The machine type, e.g. AMD64
- The underlying platform spec with as much useful information as possible. (eg. `macOS-10.16-x86_64-i386-64bit`)
- Basic `Pipeline` configuration information
- Added `Steps` and `Tasks` with `LLMs` to `Pipelines`
- Added edges between `Steps` and `Tasks`
- RuntimeErrors of Pipelines

If you have any doubts, don't hesitate to join [our Discord channel](https://discord.com/invite/hugging-face-879548962464493619) or open a GitHub issue. We'd be very happy to discuss how we can improve this.