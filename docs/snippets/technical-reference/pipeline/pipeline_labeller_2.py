from datasets import Dataset

dataset_test = Dataset.from_dict(
    {
        "input": [
            "Describe the capital of Spain in 25 words.",
            "Design a conversation between a customer and a customer service agent.",
        ],
        "generations": [
            ["Santo Domingo is the capital of Dominican Republic"],
            [
                "Customer: Hello, I'm having trouble with my purchase.\n\nCustomer Service Agent: I'm sorry to hear that. Could you please tell me more about the issue you are facing?\n\nCustomer: Yes, I ordered a pair of shoes from your company a week ago, but I haven't received them yet.\n\nCustomer Service Agent: I apologize for the inconvenience. Could you please provide me with your order number and full name so I can look into this for you?\n\nCustomer: Sure, my name is John Doe and my order number is ABCD1234.\n\nCustomer Service Agent: Thank you, John. I have checked on your order and it appears that it is still being processed. It should be shipped out within the next 24 hours.\n\nCustomer: That's good to hear, but can you also tell me the expected delivery time?\n\nCustomer Service Agent: Absolutely, based on your location, the estimated delivery time is 3-5 business days after shipping. You will receive a tracking number via email once the item is shipped, which will provide real-time updates on your package.\n\nCustomer: Thanks for the information. One more thing, what is your return policy if the shoes don't fit?\n\nCustomer Service Agent: Our company offers a 30-day return policy. If you are not satisfied with the product or if it doesn't fit, you can return it for a full refund or an exchange within 30 days of delivery. Please keep in mind that the product must be in its original packaging and in the same condition as when you received it.\n\nCustomer: Okay, that's good to know. Thank you for your help.\n\nCustomer Service Agent: You're welcome, John. I'm glad I could assist you. If you have any further questions or concerns, please don't hesitate to reach out to us. Have a great day!"
            ],
        ],
    }
)

ds_labelled = pipe_labeller.generate(dataset_test)
