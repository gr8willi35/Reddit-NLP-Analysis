import requests
from datetime import datetime
import traceback
import time
import json
import sys

username = "gr8willi35"  # put the username you want to download in the quotes
subreddit = "pokemongo"  # put the subreddit you want to download in the quotes
# leave either one blank to download an entire user's or subreddit's history
# or fill in both to download a specific users history from a specific subreddit

filter_string = None
if username == "" and subreddit == "":
	print("Fill in either username or subreddit")
	sys.exit(0)
elif username == "" and subreddit != "":
	filter_string = f"subreddit={subreddit}"
elif username != "" and subreddit == "":
	filter_string = f"author={username}"
else:
	filter_string = f"author={username}&subreddit={subreddit}"

url = "https://api.pushshift.io/reddit/{}/search?limit=1000&{}&before="

start_time = datetime.utcnow()


def downloadFromUrl(filename, object_type):
	print(f"Saving {object_type}s to {filename}")

	count = 0
	handle = open(filename, 'w')
	previous_epoch = int(start_time.timestamp())
	while True:
		new_url = url.format(object_type, filter_string)+str(previous_epoch)
		json_text = requests.get(new_url, headers={'User-Agent': "Post downloader by /u/Watchful1"})
		time.sleep(1)  # pushshift has a rate limit, if we send requests too fast it will start returning error messages
		try:
			json_data = json_text.json()
		except json.decoder.JSONDecodeError:
			time.sleep(1)
			continue

		if 'data' not in json_data:
			break
		objects = json_data['data']
		if len(objects) == 0:
			break

		for object in objects:
			previous_epoch = object['created_utc'] - 1
			count += 1
			if object_type == 'comment':
				try:
					handle.write(str(object['score']))
					handle.write(" : ")
					handle.write(datetime.fromtimestamp(object['created_utc']).strftime("%Y-%m-%d"))
					handle.write("\n")
					handle.write(object['body'].encode(encoding='ascii', errors='ignore').decode())
					handle.write("\n-------------------------------\n")
				except Exception as err:
					print(f"Couldn't print comment: https://www.reddit.com{object['permalink']}")
					print(traceback.format_exc())
			elif object_type == 'submission':
				if object['is_self']:
					if 'selftext' not in object:
						continue
					try:
						handle.write(str(object['score']))
						handle.write(" : ")
						handle.write(datetime.fromtimestamp(object['created_utc']).strftime("%Y-%m-%d"))
						handle.write("\n")
						handle.write(object['selftext'].encode(encoding='ascii', errors='ignore').decode())
						handle.write("\n-------------------------------\n")
					except Exception as err:
						print(f"Couldn't print post: {object['url']}")
						print(traceback.format_exc())

		print("Saved {} {}s through {}".format(count, object_type, datetime.fromtimestamp(previous_epoch).strftime("%Y-%m-%d")))

	print(f"Saved {count} {object_type}s")
	handle.close()


downloadFromUrl("posts.txt", "submission")
downloadFromUrl("comments.txt", "comment")
"""
What proportion of people on a subreddit delete their posts? This script pulls
from the Pushshift and Reddit APIs and generates a file with columns for
submissions' deletion status of author and message, at time of Pushshift's
indexing (often within 24 hours) and Reddit's current version.
"""

import argparse  # http://docs.python.org/dev/library/argparse.html
import datetime as dt
import logging
import numpy as np
import pandas as pd
import random
import sys
import time
from pathlib import Path, PurePath
from tqdm import tqdm

# https://www.reddit.com/dev/api/
import praw  # https://praw.readthedocs.io/en/latest

from web_api_tokens import (
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
)

# https://github.com/reagle/thunderdell
from web_utils import get_JSON

# https://github.com/pushshift/api
# import psaw  # https://github.com/dmarx/psaw no exclude:not

REDDIT = praw.Reddit(
    user_agent=REDDIT_USER_AGENT,
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
)

exception = logging.exception
critical = logging.critical
error = logging.error
warning = logging.warning
info = logging.info
debug = logging.debug


def get_reddit_info(id):
    """Given id, returns info from reddit."""

    if not args.skip:
        author = "[deleted]"
        is_deleted = False
        is_removed = False

        submission = REDDIT.submission(id=id)
        author = "[deleted]" if not submission.author else submission.author
        debug(f"{author=}")
        is_deleted = submission.selftext == "[deleted]"
        is_removed = submission.selftext == "[removed]"
    else:
        author = "NA"
        is_deleted = "NA"
        is_removed = "NA"
    return author, is_deleted, is_removed


def check_for_deleted(pushshift_results):
    """Given pushshift query results, return dataframe of info about
    submissions.
    """
    """
    https://github.com/pushshift/api
    https://github.com/dmarx/psaw
    https://www.reddit.com/dev/api/
    https://praw.readthedocs.io/en/latest
    """

    # Use these for manual confirmation of results
    PUSHSHIFT_API_URL = (
        "https://api.pushshift.io/reddit/submission/search?ids="
    )
    REDDIT_API_URL = "https://api.reddit.com/api/info/?id=t3_"

    results_checked = []
    for pr in tqdm(pushshift_results):
        debug(f"{pr['id']=} {pr['author']=} {pr['title']=}\n")
        created_utc = dt.datetime.fromtimestamp(pr["created_utc"]).strftime(
            "%Y%m%d %H:%M:%S"
        )
        elapsed_hours = round((pr["retrieved_on"] - pr["created_utc"]) / 3600)
        author_r, is_deleted_r, is_removed_r = get_reddit_info(pr["id"])
        results_checked.append(
            (
                author_r,  # author_r(eddit)
                pr["author"] == "[deleted]",  # del_author_p(ushshift)
                author_r == "[deleted]",  # del_author_r(eddit)
                pr["title"],  # title (pushshift)
                pr["id"],  # id (pushshift)
                created_utc,
                elapsed_hours,  # elapsed hours when pushshift indexed
                pr["score"],  # at time of ingest
                pr["num_comments"],  # updated as comments ingested?
                pr.get("selftext", "") == "[deleted]",  # del_text_p(ushshift)
                is_deleted_r,  # del_text_r(eddit)
                is_removed_r,  # rem_text_r(eddit)
                pr["url"],
                # PUSHSHIFT_API_URL + r["id"],
                # REDDIT_API_URL + r["id"],
            )
        )
    debug(results_checked)
    posts_df = pd.DataFrame(
        results_checked,
        columns=[
            "author_r",
            "del_author_p",  # on pushshift
            "del_author_r",  # on reddit
            "title",
            "id",
            "created_utc",
            "elapsed_hours",
            "score_p",
            "num_comments_p",
            "del_text_p",
            "del_text_r",
            "rem_text_r",
            "url",
            # "url_api_p",
            # "url_api_r",
        ],
    )
    return posts_df


def query_pushshift(
    name, limit, after, before, subreddit, query="", num_comments=">0",
):
    """Given search parameters, query pushshift and return JSON.
    # https://github.com/pushshift/api
    """

    if isinstance(after, str):
        after_human = after
    else:
        after_human = time.strftime("%Y%m%d %H:%M:%S", time.gmtime(after))
    if isinstance(before, str):
        before_human = before
    else:
        before_human = time.strftime("%Y%m%d %H:%M:%S", time.gmtime(before))
    critical(f"******* between {after_human} and {before_human}")

    optional_params = ""
    if after:
        optional_params += f"&after={after}"
    if before:
        optional_params += f"&before={before}"
    if num_comments:
        optional_params += f"&num_comments={num_comments}"
    if not args.moderated_include:
        optional_params += f"&selftext:not=[removed]"

    pushshift_url = (
        f"https://api.pushshift.io/reddit/submission/search/"
        f"?limit={limit}&subreddit={subreddit}{optional_params}"
    )
    print(f"{pushshift_url=}")
    list_of_dicts = get_JSON(pushshift_url)["data"]
    return list_of_dicts


def ordered_lin_sample(items, limit):
    """Linear sample from items with order preserved"""

    info(f"{len(items)=}")
    info(f"{limit=}")
    sampled_index = np.linspace(0, len(items) - 1, limit).astype(int).tolist()
    info(f"{sampled_index=}")
    sampled_items = [items[token] for token in sampled_index]
    return sampled_items


def ordered_random_sample(items, limit):
    """Random sample from items with order preserved"""

    index = range(len(items))
    sampled_index = sorted(random.sample(index, limit))
    info(f"{sampled_index=}")
    sampled_items = [items[token] for token in sampled_index]
    return sampled_items


def collect_pushshift_results(
    name, limit, after, before, subreddit, query="", num_comments=">0",
):
    """Pushshift limited to 100 results, so need multiple queries to
    collect results in date range up to or sampled from limit."""

    results = results_all = query_pushshift(
        name, limit, after, before, subreddit, query, num_comments
    )
    if args.sample:  # collect whole range and then sample to limit
        while len(results) != 0:
            time.sleep(1)
            after_new = results[-1]["created_utc"]  # + 1?
            results = query_pushshift(
                name, limit, after_new, before, subreddit, query, num_comments
            )
            results_all.extend(results)
        print(f"{len(results_all)=}")
        results_all = ordered_lin_sample(results_all, limit)
        print(f"returning linspace sample of size {len(results_all)}")
    else:  # collect only up to limit
        while len(results) != 0 and len(results_all) < limit:
            time.sleep(1)
            after_new = results[-1]["created_utc"]  # + 1?
            results = query_pushshift(
                name, limit, after_new, before, subreddit, query, num_comments
            )
            results_all.extend(results)
        results_all = results_all[0:limit]
        print(f"returning random sample of size {len(results_all)}")

    return results_all


def collect_pushshift_results_old(
    name, limit, after, before, subreddit, query="", num_comments=">0",
):
    """Pushshift limited to 100 results, so need multiple queries to
    collect results in date range up to limit."""

    results = results_all = query_pushshift(
        name, limit, after, before, subreddit, query, num_comments
    )
    while len(results) != 0 and len(results_all) < limit:
        time.sleep(1)
        after_new = results[-1]["created_utc"]  # + 1?
        after_new_human = time.strftime(
            "%a, %d %b %Y %H:%M:%S", time.gmtime(after_new)
        )
        info(f"****** {after_new_human=} ********")
        results = query_pushshift(
            name, limit, after_new, before, subreddit, query, num_comments
        )
        results_all.extend(results)
        debug(f"{len(results_all)=} {len(results)=}")

    return results_all[0:limit]


def export_df(name, df):

    df.to_csv(f"{name}.csv", encoding="utf-8-sig", index=False)
    print(f"saved dataframe of shape {df.shape} to '{name}.csv'")


def main(argv):
    """Process arguments"""
    arg_parser = argparse.ArgumentParser(
        description="Script for querying reddit APIs"
    )

    # optional arguments
    arg_parser.add_argument(
        "-a",
        "--after",
        type=str,
        default=False,
        help=f"""submissions after: epoch, integer[s|m|h|d], or Y-m-d""",
    )
    arg_parser.add_argument(
        "-b",
        "--before",
        type=str,
        default=False,
        help="""submissions before: epoch, integer[s|m|h|d], or Y-m-d""",
    )
    arg_parser.add_argument(
        "-k",
        "--keep",
        action="store_true",
        default=False,
        help="keep existing CSV files and don't overwrite (default: %(default)s)",
    )
    arg_parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=5,
        help="limit to (default: %(default)s) results",
    )
    arg_parser.add_argument(
        "-m",
        "--moderated_include",
        action="store_true",
        default=False,
        help="include moderated ['removed'] submissions (default: %(default)s)",
    )
    arg_parser.add_argument(
        "-n",
        "--num_comments",
        type=str,
        default=False,
        help="""number of comments threshold """
        r"""'[<>]\d+]' (default: %(default)s). """
        """Note: this is updated as Pushshift ingests, `score` is not.""",
    )
    arg_parser.add_argument(
        "-r",
        "--subreddit",
        type=str,
        default="AmItheAsshole",
        help="subreddit to query (default: %(default)s)",
    )
    arg_parser.add_argument(
        "--sample",
        action="store_true",
        default=False,
        help="""sample complete date range up to limit, rather than """
        """first submissions within limit (default: %(default)s)""",
    )
    arg_parser.add_argument(
        "--skip",
        action="store_true",
        default=False,
        help="skip reddit queries and return after pushshift (default: %(default)s)",
    )
    arg_parser.add_argument(
        "-L",
        "--log-to-file",
        action="store_true",
        default=False,
        help="log to file %(prog)s.log",
    )
    arg_parser.add_argument(
        "-V",
        "--verbose",
        action="count",
        default=0,
        help="increase logging verbosity (specify multiple times for more)",
    )
    arg_parser.add_argument("--version", action="version", version="0.3")
    args = arg_parser.parse_args(argv)

    log_level = logging.ERROR  # 40

    if args.verbose == 1:
        log_level = logging.WARNING  # 30
    elif args.verbose == 2:
        log_level = logging.INFO  # 20
    elif args.verbose >= 3:
        log_level = logging.DEBUG  # 10
    LOG_FORMAT = "%(levelname).3s %(funcName).5s: %(message)s"
    if args.log_to_file:
        print("logging to file")
        logging.basicConfig(
            filename=f"{str(PurePath(__file__).name)}.log",
            filemode="w",
            level=log_level,
            format=LOG_FORMAT,
        )
    else:
        logging.basicConfig(level=log_level, format=LOG_FORMAT)

    return args


if __name__ == "__main__":
    args = main(sys.argv[1:])

    # syntactical tweaks to filename
    if args.after and args.before:
        date = f"{args.after.replace('-','')}-{args.before.replace('-','')}"
    elif args.after:
        date = f"{args.after.replace('-','')}-NOW"
    elif args.before:
        date = f"THEN-{args.before.replace('-','')}"
    if args.num_comments:
        num_comments = args.num_comments
        if num_comments[0] == ">":
            num_comments = num_comments[1:] + "+"
        elif num_comments[0] == "<":
            num_comments = num_comments[1:] + "-"
        num_comments = "n" + num_comments
    else:
        num_comments = "n_"
    if args.sample:
        sample = "_sampled"
    else:
        sample = ""

    queries = (
        {
            "name": (
                f"""reddit_{date}_{args.subreddit}_{num_comments}"""
                f"""_l{args.limit}{sample}"""
            ),
            "limit": args.limit,
            "before": args.before,
            "after": args.after,
            "subreddit": args.subreddit,
            "num_comments": args.num_comments,
        },
    )

    for query in queries:
        print(f"{query=}")
        if args.keep and Path(f"{query['name']}.csv").exists():
            debug(f"{query['name']}.csv already exists")
            continue
        else:
            ps_results = collect_pushshift_results(**query)
            posts_df = check_for_deleted(ps_results)
            export_df(query["name"], posts_df)