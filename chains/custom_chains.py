from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate


from output_parsers import summary_parser, profilesearcher_parser, topics_of_interest_parser

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
llm_creative = ChatOpenAI(temperature=1, model_name="gpt-3.5-turbo")


def get_summary_chain() -> RunnableSequence:
    summary_template = """
         given the information about a person from linkedin {information}, and twitter posts {twitter_posts} I want you to create:
         1. a short summary
         2. two interesting facts about them
         \n{format_instructions}
     """

    summary_prompt_template = PromptTemplate(
        input_variables=["information", "twitter_posts"],
        template=summary_template,
        partial_variables={
            "format_instructions": summary_parser.get_format_instructions()
        },
    )

    return summary_prompt_template | llm | summary_parser


def get_interests_chain() -> RunnableSequence:
    interesting_facts_template = """
         given the information about a person from linkedin {information}, and twitter posts {twitter_posts} I want you to create:
         3 topics that might interest them
        \n{format_instructions}
     """

    interesting_facts_prompt_template = PromptTemplate(
        input_variables=["information", "twitter_posts"],
        template=interesting_facts_template,
        partial_variables={
            "format_instructions": topics_of_interest_parser.get_format_instructions()
        },
    )

    return interesting_facts_prompt_template | llm | topics_of_interest_parser


def get_profilesearcher_chain() -> RunnableSequence:
    profilesearcher_template = """
         given the information about a person from linkedin {information}, and twitter posts {twitter_posts} I want you to create:
         2 creative Ice breakers with them that are derived from their activity on Linkedin and twitter, preferably on latest tweets
        \n{format_instructions}
     """

    profilesearcher_prompt_template = PromptTemplate(
        input_variables=["information", "twitter_posts"],
        template=profilesearcher_template,
        partial_variables={
            "format_instructions": profilesearcher_parser.get_format_instructions()
        },
    )

    return profilesearcher_prompt_template | llm | profilesearcher_parser
