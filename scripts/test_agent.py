import asyncio
from scripts.fl_agent_tool import step
from scripts.fl_datagen import make_prompt

async def test_agent_responses():
    # Test prompt
    grid_str = "\nF F F F\nS F H G\nF F H F\nH F H F"
    observation = make_prompt(grid_str)
    
    # Mock some model responses to test parsing
    test_responses = [
        "<answer>UP</answer>",
        "<simulate>UP DOWN RIGHT</simulate>", 
        "<answer>invalid_action</answer>",
        "random text with no tags",
        "<simulate>UP DOWN INVALID_ACTION</simulate>",
        "<simulate>UP RIGHT RIGHT RIGHT DOWN</simulate>"
    ]
    
    for i, response in enumerate(test_responses):
        print(f"\n--- Test {i+1} ---")
        print(f"Response: {response}")
        result = await step(observation, response, None)
        print(f"Result: {result}")

asyncio.run(test_agent_responses()) 