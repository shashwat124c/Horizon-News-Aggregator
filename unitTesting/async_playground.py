import asyncio
import time

# -------------------------------------------------------------
# Example 1: Synchronous (Blocking) Example
# -------------------------------------------------------------
def make_tea_sync():
    print("[Sync] Boiling water...")
    time.sleep(2)  # Blocks the entire program (no other code can run)
    print("[Sync] Water boiled!")

def toast_bread_sync():
    print("[Sync] Toasting bread...")
    time.sleep(2)  # Blocks the entire program
    print("[Sync] Bread toasted!")

def run_sync_breakfast():
    start = time.time()
    print("--- Starting Synchronous Breakfast ---")
    make_tea_sync()
    toast_bread_sync()
    print(f"Total time taken: {time.time() - start:.2f} seconds\n")


# -------------------------------------------------------------
# Example 2: Asynchronous (Non-Blocking) Example
# -------------------------------------------------------------
async def make_tea_async():
    print("[Async] Boiling water started...")
    # asyncio.sleep yields control back to the event loop.
    # While this sleeps, other async tasks can run!
    await asyncio.sleep(2)
    print("[Async] Water boiled!")

async def toast_bread_async():
    print("[Async] Toasting bread started...")
    await asyncio.sleep(2)
    print("[Async] Bread toasted!")

async def run_async_breakfast():
    start = time.time()
    print("--- Starting Asynchronous Breakfast ---")
    # asyncio.gather runs both tasks concurrently
    await asyncio.gather(
        make_tea_async(),
        toast_bread_async()
    )
    print(f"Total time taken: {time.time() - start:.2f} seconds\n")


# -------------------------------------------------------------
# Example 3: Understanding why 'await' is needed
# -------------------------------------------------------------
async def test_coroutine():
    print("This is a coroutine!")

def run_coroutine_explanation():
    print("--- Coroutine Object Demonstration ---")
    # Calling an async function does NOT run it.
    # It just returns a "coroutine object".
    coro = test_coroutine()
    print(f"Returned object type: {type(coro)}")
    print("Notice that 'This is a coroutine!' was NOT printed yet.")
    
    # We must run it with asyncio.run() or await it inside another async function
    print("Now we run it properly using asyncio.run:")
    asyncio.run(coro)


if __name__ == "__main__":
    # 1. Run the blocking synchronous flow (Takes ~4 seconds total)
    run_sync_breakfast()
    
    # 2. Run the concurrent asynchronous flow (Takes ~2 seconds total)
    asyncio.run(run_async_breakfast())
    
    # 3. Understand coroutine objects
    run_coroutine_explanation()
