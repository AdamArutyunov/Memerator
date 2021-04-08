from telegram.ext import Updater, MessageHandler, Filters
from telegram.ext import CallbackContext, CommandHandler
from telegram import ReplyKeyboardMarkup
from Constants import *

reply_keyboard = [['/generate', '/recognize']]
markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=False)


def start(update, context):
    update.message.reply_text("Hello, I am Memerator! I can recognize and generate memes", reply_markup=markup)


def recognize(update, context):
    update.message.reply_text("АХАХАХАХАХА ВАХ КАКОЙ СМЕШНОЙ МЕМ Я АЖ ОБКЕКАЛСЯ", reply_markup=markup)


def generate(update, context):
    # update.message.reply_photo(photo)
    update.message.reply_text("мем")


def error(update, context):
    update.message.reply_text("Я понимаю только язык МЕМОВ")

def karelia(update, context):
    update.message.reply_text("Вы нашли пасхалку. Хайль Карелия!")

    
def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("recognize", recognize))
    dp.add_handler(CommandHandler("generate", generate))
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("karelia", karelia))
    dp.add_handler(MessageHandler(Filters.text, error))

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()