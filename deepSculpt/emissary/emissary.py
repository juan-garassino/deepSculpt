from deepSculpt.emissary.tools.client import getCreds
from deepSculpt.emissary.tools.instagram import (
    createMediaObject,
    getMediaObjectStatus,
    publishMedia,
    getContentPublishingLimit,
)

if __name__ == "__main__":
    params = getCreds()  # get creds from defines

    params["media_type"] = "IMAGE"  # type of asset
    params[
        "media_url"
    ] = "https://justinstolpe.com/sandbox/ig_publish_content_img.png"  # url on public server for the post
    params[
        "caption"
    ] = "This image was posted through the Instagram Graph API with a python script I wrote! Go check out the video tutorial on my YouTube channel."
    params["caption"] += "\n."
    params["caption"] += "\nyoutube.com/justinstolpe"
    params["caption"] += "\n."
    params[
        "caption"
    ] += "\n#instagram #graphapi #instagramgraphapi #code #coding #programming #python #api #webdeveloper #codinglife #developer #coder #tech #developerlife #webdev #youtube #instgramgraphapi"  # caption for the post

    imageMediaObjectResponse = createMediaObject(
        params
    )  # create a media object through the api
    imageMediaObjectId = imageMediaObjectResponse["json_data"][
        "id"
    ]  # id of the media object that was created
    imageMediaStatusCode = "IN_PROGRESS"

    print("\n---- IMAGE MEDIA OBJECT -----\n")  # title
    print("\tID:")  # label
    print(" " + imageMediaObjectId)  # id of the object

    while (
        imageMediaStatusCode != "FINISHED"
    ):  # keep checking until the object status is finished
        imageMediaObjectStatusResponse = getMediaObjectStatus(
            imageMediaObjectId, params
        )  # check the status on the object
        imageMediaStatusCode = imageMediaObjectStatusResponse["json_data"][
            "status_code"
        ]  # update status code

        print("\n---- IMAGE MEDIA OBJECT STATUS -----\n")  # display status response
        print("\tStatus Code:")  # label
        print(" " + imageMediaStatusCode)  # status code of the object

        time.sleep(5)  # wait 5 seconds if the media object is still being processed

    publishImageResponse = publishMedia(
        imageMediaObjectId, params
    )  # publish the post to instagram

    print("\n---- PUBLISHED IMAGE RESPONSE -----\n")  # title
    print("\tResponse:")  # label
    print(publishImageResponse["json_data_pretty"])  # json response from ig api

    params["media_type"] = "VIDEO"  # type of asset
    params[
        "media_url"
    ] = "https://justinstolpe.com/sandbox/ig_publish_content_vid.mp4"  # url on public server for the post
    params[
        "caption"
    ] = "This video was posted through the Instagram Graph API with a python script I wrote! Go check out the video tutorial on my YouTube channel."
    params["caption"] += "\n."
    params["caption"] += "\nyoutube.com/justinstolpe"
    params["caption"] += "\n."
    params[
        "caption"
    ] += "\n#instagram #graphapi #instagramgraphapi #code #coding #programming #python #api #webdeveloper #codinglife #developer #coder #tech #developerlife #webdev #youtube #instgramgraphapi"  # caption for the post

    videoMediaObjectResponse = createMediaObject(
        params
    )  # create a media object through the api
    videoMediaObjectId = videoMediaObjectResponse["json_data"][
        "id"
    ]  # id of the media object that was created
    videoMediaStatusCode = "IN_PROGRESS"

    print("\n---- VIDEO MEDIA OBJECT -----\n")  # title
    print("\tID:")  # label
    print(" " + videoMediaObjectId)  # id of the object

    while (
        videoMediaStatusCode != "FINISHED"
    ):  # keep checking until the object status is finished
        videoMediaObjectStatusResponse = getMediaObjectStatus(
            videoMediaObjectId, params
        )  # check the status on the object
        videoMediaStatusCode = videoMediaObjectStatusResponse["json_data"][
            "status_code"
        ]  # update status code

        print("\n---- VIDEO MEDIA OBJECT STATUS -----\n")  # display status response
        print("\tStatus Code:")  # label
        print(" " + videoMediaStatusCode)  # status code of the object

        time.sleep(5)  # wait 5 seconds if the media object is still being processed

    publishVideoResponse = publishMedia(
        videoMediaObjectId, params
    )  # publish the post to instagram

    print("\n---- PUBLISHED IMAGE RESPONSE -----\n")  # title
    print("\tResponse:")  # label
    print(publishVideoResponse["json_data_pretty"])  # json response from ig api

    contentPublishingApiLimit = getContentPublishingLimit(
        params
    )  # get the users api limit

    print("\n---- CONTENT PUBLISHING USER API LIMIT -----\n")  # title
    print("\tResponse:")  # label
    print(contentPublishingApiLimit["json_data_pretty"])  # json response from ig api
