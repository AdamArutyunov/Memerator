from telegram.ext import Updater, MessageHandler, Filters
from telegram.ext import CallbackContext, CommandHandler
from telegram import ReplyKeyboardMarkup
import requests


TOKEN = "c"
reply_keyboard = [['/generate', '/recognize']]
markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=False)


def start(update, context):
    update.message.reply_text("Hello, I am Memerator! I can recognize and generate memes", reply_markup=markup)


def recognize(update, context):
    '''# update.message.reply_text(requests.get("http://134.209.94.3:5000/recognize/im.jpg"))
    url = 'http://134.209.94.3:5000/recognize'
    fp = open('im.jpg', 'rb')
    files = {'image': fp}

    update.message.reply_text(requests.post(url, files=files))
    fp.close()'''

    photo_path = update.message.photo[-1].get_file().download()
    fp = open(photo_path, "rb")
    files = {"image": fp}
    resp = requests.post("http://134.209.94.3:5000/recognize", files=files).json()
    update.message.reply_text(f"Meme: {resp['meme']}\nProbability: {resp['probability']}")
    fp.close()
    


def generate(update, context):
    photo = requests.get("http://134.209.94.3:5000/generate")

    with open("image.png", "wb") as f:
        f.write(photo.content)

    update.message.reply_photo(open("image.png", "rb"))


def error(update, context):
    update.message.reply_text("I am only undersand the MEME language!")

def helpp(update, context):
    update.message.reply_text("Send me a message to get the information about your meme!")

def karelia(update, context):
    update.message.reply_text("Вы нашли пасхалку. Хайль Карелия! Voiten Kunnia! Слава Победе!")

    
def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(MessageHandler(Filters.photo, recognize))
    dp.add_handler(CommandHandler("generate", generate))
    dp.add_handler(CommandHandler("recognize", helpp))
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("karelia", karelia))
    dp.add_handler(MessageHandler(Filters.text, error))

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
