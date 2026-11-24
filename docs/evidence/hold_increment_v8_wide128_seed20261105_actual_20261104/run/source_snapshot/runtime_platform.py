"""Keep the original Windows renderer workaround out of Linux launches."""
import os

def platform_kit_args(arguments: str) -> str:
    if os.name != "nt":
        arguments = arguments.replace("--/app/vulkan=false", "--/app/vulkan=true")
    return arguments
