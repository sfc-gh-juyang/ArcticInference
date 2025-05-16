
import logging

# This is to avoid logs for tasks that were already cancelled
def exception_handler(loop, context):
    exception = context["exception"]
    if isinstance(exception, StopAsyncIteration):
        pass
    else:
        message = context["message"]
        logging.error(f"Task failed, msg={message}, exception={exception}")

