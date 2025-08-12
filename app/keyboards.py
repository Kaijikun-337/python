from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

main = ReplyKeyboardMarkup(keyboard=
    [
        [
            KeyboardButton(text='Chat'),
            KeyboardButton(text='Exit')
        ]                               
    ],
    resize_keyboard=True, input_field_placeholder='Choose the options' 
)