print(pipe_dataset["input"][-1])
# Create a 3 turn conversation between a customer and a grocery store clerk - that is, 3 per person. Then tell me what they talked about.

print(pipe_dataset["generations"][-1][-1])
# Customer: Hi there, I'm looking for some fresh berries. Do you have any raspberries or blueberries in stock?

# Grocery Store Clerk: Yes, we have both raspberries and blueberries in stock today. Would you like me to grab some for you or can you find them yourself?

# Customer: I'd like your help getting some berries. Can you also suggest which variety is sweeter? Raspberries or blueberries?

# Grocery Store Clerk: Raspberries and blueberries both have distinct flavors. Raspberries are more tart and a little sweeter whereas blueberries tend to be a little sweeter and have a milder taste. It ultimately depends on your personal preference. Let me grab some of each for you to try at home and see which one you like better.

# Customer: That sounds like a great plan. How often do you receive deliveries? Do you have some new varieties of berries arriving soon?

# Grocery Store Clerk: We receive deliveries twice a week, on Wednesdays and Sundays. We also have a rotation of different varieties of berries throughout the season, so keep an eye out for new arrivals. Thanks for shopping with us, can I help you with anything else today?

# Customer: No, that's all for now. I'm always happy to support your local store.

# turn 1: berries, fresh produce availability, customer preference
# turn 2: product recommendations based on taste and personal preference, availability
# turn 3: store acknowledgment, shopping gratitude, loyalty and repeat business expectation.

print(pipe_dataset["rating"][-1][-1])
# 5.0

print(pipe_dataset["rationale"][-1][-1])
# The text accurately follows the given instructions and provides a conversation between a customer and a grocery store clerk. The information provided is correct, informative, and aligned with the user's intent. There are no hallucinations or misleading details.
