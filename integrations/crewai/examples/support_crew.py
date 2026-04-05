"""
Example: TraceRazor + CrewAI customer support crew.

pip install tracerazor-crewai[crewai]
export TRACERAZOR_BIN=/path/to/TraceRazor/target/release/tracerazor
"""

from crewai import Agent, Task, Crew
from tracerazor_crewai import TraceRazorCallback

callback = TraceRazorCallback(
    agent_name="support-crew",
    threshold=70,
)

researcher = Agent(
    role="Support Researcher",
    goal="Look up order details and refund eligibility",
    backstory="Expert at retrieving order information quickly.",
)
resolver = Agent(
    role="Support Resolver",
    goal="Process refunds and close tickets",
    backstory="Handles refunds with empathy and accuracy.",
)

research_task = Task(
    description="Look up order ORD-9182 and check refund eligibility.",
    agent=researcher,
)
resolve_task = Task(
    description="Process the refund for ORD-9182 and confirm with the customer.",
    agent=resolver,
)

crew = Crew(
    agents=[researcher, resolver],
    tasks=[research_task, resolve_task],
    callbacks=[callback],
)

crew.kickoff()

# Analyse after the crew finishes.
report = callback.analyse()
print(report.markdown())

# CI/CD gate — raises AssertionError if TAS < threshold:
callback.assert_passes()
