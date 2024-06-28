import json
import logging
import torch

from telegram import Update, ForceReply, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from PIL import Image
from torchvision import transforms

# Включите ведение журнала
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Словарь с русскими названиями классов
RUSSIAN_CLASS_NAMES = {
    "Baked Potato": "Печёная картошка",
    "Burger": "Бургер",
    "Crispy Chicken": "Хрустящая курочка",
    "Donut": "Пончик",
    "Fries": "Картофель фри",
    "Hot Dog": "Хот-дог",
    "Pizza": "Пицца",
    "Sandwich": "Сэндвич",
    "Taco": "Тако",
    "Taquito": "Бурито",
}
# Словарь с локализациями
LOCALES = {
    'ru': {
        'start': "Привет, {user}! Я бот для обработки картинок с едой. "
                 "Отправь мне картинку, и я скажу тебе, что на ней.",
        'image_received': "Крутая картинка! Я пока не умею анализировать изображения, но скоро научусь!",
        'language_set': "Язык изменен на русский.",
        'non_command_message': "Для использования бота отправьте изображение.",
        'choose_language': "Выберите язык:",
        'help': "Данный бот создан на основе BotFather. Используя машинное зрение и нейронные сети, "
                        "он способен определить вид еды по фото. Для получения результата достаточно отправить картинку боту.",
        'image_prediction': "На изображении, вероятно, {predicted_class}.",
        'image_processing_error': "Произошла ошибка при обработке изображения.",
        'send_jpeg_image': "Пожалуйста, отправьте изображение в формате JPEG."
    },
    'en': {
        'start': "Hi, {user}! I'm a bot for processing images of food. "
                 "Send me a picture, and I'll tell you what's on it.",
        'image_received': "Cool picture! I don't know how to analyze images yet, but I will soon!",
        'language_set': "Language changed to English.",
        'non_command_message': "To use the bot, please send an image.",
        'choose_language': "Choose your language:",
        'help': "This bot is built based on BotFather. Using computer vision and neural networks,"
                        " it can identify food types from photos. To get a result, simply send a picture to the bot.",
        'image_prediction': "The image probably shows {predicted_class}.",
        'image_processing_error': "An error occurred while processing the image.",
        'send_jpeg_image': "Please send an image in JPEG format."
    },
}

# Функция для получения локализованной строки
def get_locale_string(language_code: str, key: str, **kwargs) -> str:
    """Возвращает локализованную строку по ключу и языку."""
    return LOCALES.get(language_code, LOCALES['en']).get(key, '').format(**kwargs)


# Определите функцию, которая будет вызываться при получении команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет приветственное сообщение при получении команды /start."""
    user = update.effective_user
    language_code = context.user_data.get('language_code', 'en')

    await update.message.reply_html(
        get_locale_string(language_code, 'start', user=user.mention_html()),
        reply_markup=ForceReply(selective=True),
    )

# Обработчик команды /help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет сообщение с описанием бота."""
    language_code = context.user_data.get('language_code', 'en')
    await update.message.reply_text(get_locale_string(language_code, 'help'))

# Обработчик команды /changeLng
async def change_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Изменяет язык пользователя."""
    language_code = context.user_data.get('language_code', 'en')

    # Создаем клавиатуру с выбором языка
    keyboard = ReplyKeyboardMarkup(
        [['Русский', 'English']],
        resize_keyboard=True,
        one_time_keyboard=True,
    )

    await update.message.reply_text(
        get_locale_string(language_code, 'choose_language'),
        reply_markup=keyboard,
    )


# Обработчик выбора языка
async def set_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Устанавливает выбранный язык пользователя."""
    user_choice = update.message.text

    global class_names  # Объявляем class_names как глобальную переменную

    if user_choice == 'Русский':
        context.user_data['language_code'] = 'ru'
        response_text = get_locale_string('ru', 'language_set')
        class_names = [RUSSIAN_CLASS_NAMES.get(name, name) for name in class_names] # Заменяем на русские названия
    else:
        context.user_data['language_code'] = 'en'
        response_text = get_locale_string('en', 'language_set')
        with open("class_names.json", 'r') as f:  # Снова загружаем английские названия
            class_names = json.load(f)

    await update.message.reply_text(
        response_text,
        reply_markup=ReplyKeyboardMarkup([['/start']])  # Возвращаем клавиатуру по умолчанию
    )


# Функция для обработки изображения (вызывается и для фото, и для документов)
async def process_image(update: Update, context: ContextTypes.DEFAULT_TYPE, image_path: str) -> None:
    """Обрабатывает изображение и отправляет результат пользователю."""
    language_code = context.user_data.get('language_code', 'en')

    try:
        # Загружаем изображение с помощью PIL
        image = Image.open(image_path).convert("RGB")

        # Преобразование и предсказание
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            _, preds = torch.max(outputs, 1)

        # Получаем предсказанный класс
        predicted_class = class_names[preds[0]]

        # Отправляем результат пользователю
        await update.message.reply_text(
            get_locale_string(language_code, 'image_prediction', predicted_class=predicted_class),
            reply_to_message_id=update.message.message_id
        )


    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {e}")
        await update.message.reply_text(
            get_locale_string(language_code, 'image_processing_error')
        )

# Обработчик изображений
async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает отправленные изображения."""
    #language_code = context.user_data.get('language_code', 'en')

    file = await context.bot.get_file(update.message.photo[-1].file_id)
    image_path = "temp_image.jpg"
    await file.download_to_drive(image_path)
    await process_image(update, context, image_path)
# Обработчик документов (JPEG)
async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает JPEG, отправленные как документы."""
    language_code = context.user_data.get('language_code', 'en')

    if update.message.document.mime_type == 'image/jpeg':
        file = await context.bot.get_file(update.message.document.file_id)
        image_path = "temp_image.jpg"
        await file.download_to_drive(image_path)

        await process_image(update, context, image_path) # Вызываем функцию для обработки

    else:
        await update.message.reply_text(
            get_locale_string(language_code, 'send_jpeg_image')
        )
# Обработчик некомандных сообщений и изображений
async def non_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает сообщения, не являющиеся командами."""
    language_code = context.user_data.get('language_code', 'en')
    await update.message.reply_text(get_locale_string(language_code, 'non_command_message'))

# Функция загрузки модели и имен классов
def load_model_and_classes(model_path="Fast_Food_Classification_V2_v2.pt",
                           class_names_path="class_names.json"):
    """Загружает модель и имена классов из файлов."""
    try:
        global model, class_names, device, preprocess

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(model_path)
        model.to(device)

        # Преобразование изображения
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        return model, class_names
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели или имен классов: {e}")
        return None, None
def main() -> None:
    """Запускает бота."""
    application = Application.builder().token("7387262541:AAHKzEGN5l-cRUqlwdlPBXpSHknQeztAd8g").build()

    model, class_names = load_model_and_classes()
    if model is None or class_names is None:
        logger.error("Не удалось загрузить модель или имена классов. Выход.")
        return

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("change_lng", change_language))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex(r'^(Русский|English)$'), set_language))
    application.add_handler(MessageHandler(filters.PHOTO, image_handler))
    application.add_handler(MessageHandler(filters.Document.JPG, document_handler))

    application.add_handler(MessageHandler(~filters.COMMAND, non_command_handler))

    application.run_polling()


if __name__ == "__main__":
    main()