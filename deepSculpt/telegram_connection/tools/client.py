import requests
import json
import os


def run():
    pass


def getCreds():
    """
    Returns:
            dictonary: credentials needed globally
    """

    creds = dict()  # dictionary to hold everything

    creds["access_token"] = os.environ.get(
        "ACCESS_TOKEN"
    )  # access token for use with all api calls
    creds["client_id"] = os.environ.get(
        "INSTAGRAM_APP_ID"
    )  # client id from facebook app IG Graph API Test
    creds["client_secret"] = os.environ.get(
        "FB-APP-CLIENT-SECRET"
    )  # client secret from facebook app
    creds["page_id"] = os.environ.get("INSTAGRAM_APP_NAME")  # users page id
    creds["instagram_account_id"] = os.environ.get(
        "INSTAGRAM-BUSINESS-ACCOUNT-ID"
    )  # users instagram account id
    creds["ig_username"] = os.environ.get("INSTAGRAM_USER_NAME")  # ig username

    creds["graph_domain"] = os.environ.get(
        "https://graph.facebook.com/"
    )  # base domain for api calls
    creds["graph_version"] = os.environ.get("v6.0")  # version of the api we are hitting

    creds["endpoint_base"] = (
        creds["graph_domain"] + creds["graph_version"] + "/"
    )  # base endpoint with domain and version

    creds["debug"] = "no"  # debug mode for api call

    return creds


def makeApiCall(url, endpointParams, debug="no"):
    """Request data from endpoint with params

    Args:
            url: string of the url endpoint to make request from
            endpointParams: dictionary keyed by the names of the url parameters
    Returns:
            object: data from the endpoint
    """

    data = requests.get(url, endpointParams)  # make get request

    response = dict()  # hold response info
    response["url"] = url  # url we are hitting
    response["endpoint_params"] = endpointParams  # parameters for the endpoint
    response["endpoint_params_pretty"] = json.dumps(
        endpointParams, indent=4
    )  # pretty print for cli
    response["json_data"] = json.loads(data.content)  # response data from the api
    response["json_data_pretty"] = json.dumps(
        response["json_data"], indent=4
    )  # pretty print for cli

    if "yes" == debug:  # display out response info
        displayApiCallData(response)  # display response

    return response  # get and return content


def displayApiCallData(response):
    """Print out to cli response from api call"""

    print("\nURL: ")  # title
    print(response["url"])  # display url hit
    print("\nEndpoint Params: ")  # title
    print(response["endpoint_params_pretty"])  # display params passed to the endpoint
    print("\nResponse: ")  # title
    print(response["json_data_pretty"])  # make look pretty for cli
