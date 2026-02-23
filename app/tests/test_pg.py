# test_pg.py
import asyncio
import asyncpg

async def main():
    conn = await asyncpg.connect("postgres://postgres:oggy@localhost:5433/wizsolar")
    result = await conn.fetch("SELECT 1;")
    print(result)
    await conn.close()

asyncio.run(main())