from instabot import Bot

bot = Bot()

bot.login(username="artista.artificial", password="de@dMau33")

bot.upload_photo(
    "path",
    caption="caption",
)
