import asyncio
import logging
import aio_pika
import codefast as cf
from aio_pika import logger

logger.setLevel(logging.INFO)

async def publish(amqpurl:str, queue_name:str, message:str) -> None:
    assert isinstance(message, str)
    connection = await aio_pika.connect_robust(
        amqpurl
    )
    async with connection:
        routing_key = queue_name
        channel = await connection.channel()
        return await channel.default_exchange.publish(
            aio_pika.Message(message.encode()),
            routing_key=routing_key,
        )


async def consume(amqpurl:str, queue_name:str, _callback: callable) -> None:
    logging.basicConfig(level=logging.DEBUG)
    connection = await aio_pika.connect_robust(amqpurl)
    queue_name = queue_name

    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=10)

        queue = await channel.declare_queue(queue_name, auto_delete=False)

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    try:
                        cf.info(f'consuming {message.body}')
                        await _callback(message.body)
                    except Exception as e:
                        import traceback
                        cf.error(
                            {
                                'error': 'consume failed',
                                'exception': str(e),
                                'traceback': traceback.format_exc(),
                            }
                        )


async def main():
    await asyncio.gather(
        publish(),
        # consume(),
    )


if __name__ == "__main__":
    asyncio.run(main())
