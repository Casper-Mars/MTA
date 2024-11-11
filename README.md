# MTA

> This is a sample project of multi-agent collaboration implemented based on the LangGraph framework.

## framework

This package defines the basic framework of multi-agent collaboration. It includes the following components:

- Supervisor: A Agent that decides which worker should invoke the next action.
- Worker: A Agent that can invoke actions.
- Intention Summarizer: A Agent that can summarize the intentions of the User according to the chat history.
- Final Summarizer: A Agent that can summarize the final response according to all Worker responses.

## instance

This package provides a sample example of multi-agent collaboration. It provides some worker:

- FileLoader: A Worker that can load a file from the local file system.
- GeneralAgent: A general chat Agent that can chat with the User.
- Usecase: A Worker that handles the User's request about the use case of the system.