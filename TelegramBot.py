from telegram.ext import Updater, MessageHandler, Filters
from telegram.ext import CallbackContext, CommandHandler
from telegram import ReplyKeyboardMarkup
from Constants import *
import requests

reply_keyboard = [['/generate', '/recognize']]
markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=False)

meme_full_names = {
    "lol": "LOL",
    "megusta": "Me Gusta",
    "okay": "Okay",
    "pokerface": "Pokerface",
    "trollface": "Trollface",
    "roundface": "Round Face",
    "rufkm": "Are You Fucking Kidding Me?",
    "yaomin": "Yao Min",
    "udontsay": "You Don't Say"
}

help_string = '''
Available commands:
/generate — to generate new unique classical meme.
/recognize — to recognize classical meme from your image.
/help — to print this reminder.

Creators: @adam_arutyunov (cdarr.ru) and @johtai.'''


def start(update, context):
    message = "Hello, I am Memerator! I can recognize and generate classical memes.\n\n"
    message += help_string

    update.message.reply_text(message, reply_markup=markup)


def recognize(update, context):
    photo_path = update.message.photo[-1].get_file().download()
    photo = open(photo_path, "rb")

    files = {"image": photo}
    response = requests.post(TELEGRAM_API_ENDPOINT + "/recognize", files=files).json()

    message = f"I think it is the <b>{meme_full_names[response['meme']]}</b> meme!\n"
    message += f"I am <b>{int(response['probability'] * 100)}%</b> sure."

    update.message.reply_text(message, parse_mode="HTML")

    photo.close()
    

def generate(update, context):
    photo = requests.get(TELEGRAM_API_ENDPOINT + "/generate")

    with open("image.png", "wb") as f:
        f.write(photo.content)

    update.message.reply_photo(open("image.png", "rb"))


def error(update, context):
    update.message.reply_text("I understand only the MEME language!")


def print_help(update, context):
    update.message.reply_text(help_string)


def recognize_text(update, context):
    update.message.reply_text("OK, now send me an image that you want to be recognized.")


def karelia(update, context):
    update.message.reply_text("Вы нашли пасхалку. Хайль Карелия! Voiten Kunnia! Слава Победе!")


def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(MessageHandler(Filters.photo, recognize))

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("generate", generate))
    dp.add_handler(CommandHandler("recognize", recognize_text))
    dp.add_handler(CommandHandler("help", print_help))
    dp.add_handler(CommandHandler("karelia", karelia))

    dp.add_handler(MessageHandler(Filters.text, error))

    updater.start_polling()

    print("Bot started.")
    updater.idle()


if __name__ == '__main__':
    main()
