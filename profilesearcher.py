from typing import Tuple
from agents.linkedin_agent import lookup as linkedin_agent
from agents.twitter_agent import lookup as twitter_agent
from chains.custom_chains import (
    get_summary_chain,
    get_interests_chain,
    get_profilesearcher_chain,
)
from third_parties.linkedin import scrape_linkedin_profile
from third_parties.twitter import scrape_user_tweets, scrape_user_tweets_mock
from output_parsers import (
    Summary,
    profilesearcher,
    TopicOfInterest,
)


def profilesearcher_with(
    name: str,
) -> Tuple[Summary, TopicOfInterest, IceBreaker, str]:
    linkedin_username = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username)

    twitter_username = twitter_lookup_agent(name=name)
    tweets = scrape_user_tweets(username=twitter_username)

    summary_chain = get_summary_chain()
    summary_and_facts: Summary = summary_chain.invoke(
        input={"information": linkedin_data, "twitter_posts": tweets},
    )

    interests_chain = get_interests_chain()
    interests: TopicOfInterest = interests_chain.invoke(
        input={"information": linkedin_data, "twitter_posts": tweets}
    )

    profilesearcher_chain = get_profilesearcher_chain()
    profilesearcher: profilesearcher = profilesearcher_chain.invoke(
        input={"information": linkedin_data, "twitter_posts": tweets}
    )

    return (
        summary_and_facts,
        interests,
        profilesearcher,
        linkedin_data.get("profile_pic_url"),
    )


if __name__ == "__main__":
    pass
