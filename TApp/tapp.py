


import telebot
from firebase_admin import credentials, firestore, initialize_app
from telebot import types
from datetime import datetime

# --------------------------- Configuration ---------------------------

# Telegram bot token
TELEGRAM_BOT_TOKEN = ""

# Path to your Firebase service account key JSON file
FIREBASE_SERVICE_ACCOUNT_KEY = ""

# Live Feed Link
LIVE_FEED_LINK = "http://192.168.29.52:8000/index/"  # Replace with your actual live feed link
# Initialize Firebase
cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_KEY)
initialize_app(cred)
db = firestore.client()

# Initialize Telegram Bot
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# Dictionary to keep track of user states
user_state = {}

# --------------------------- Helper Functions ---------------------------

def send_options(user_id):
    """
    Sends the options keyboard to the user.
    """
    markup = types.ReplyKeyboardMarkup(row_width=2, one_time_keyboard=True)
    btn1 = types.KeyboardButton('/live_feed')
    btn2 = types.KeyboardButton('/attendance')
    markup.add(btn1, btn2)
    bot.send_message(user_id, "Please choose an option:", reply_markup=markup)

def validate_date_format(date_text):
    """
    Validates that the input date is in YYYY-MM-DD format.
    """
    try:
        datetime.strptime(date_text, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def fetch_attendance(date_str):
    """
    Fetches the list of present and absent students for a given date from Firestore.

    Firestore Structure:
    - Collection: 'students'
        - Document ID: {pin_number}
            - Fields: 'name', 'status', 'timestamp'
            - Subcollection: 'attendance'
                - Document ID: {YYYY-MM-DD-HH:MM:SS} (contains date_str within)
                    - Fields: 'status', 'timestamp'
    """
    try:
        present_students = {}
        absent_students = {}
        
        # Extract the date portion (YYYY-MM-DD) from the date_str
        date_part = date_str.split('-')[:3]  # ['2024', '09', '16']
        formatted_date = '-'.join(date_part)  # '2024-09-16'
        
        # Reference to the 'students' collection
        students_ref = db.collection('students')
        students_docs = students_ref.stream()
        
        for student_doc in students_docs:
            pin_number = student_doc.id
            student_data = student_doc.to_dict()
            attendance_ref = students_ref.document(pin_number).collection('attendance')
            
            # Retrieve all documents in the attendance subcollection
            attendance_docs = attendance_ref.stream()
            found_attendance = False  # Flag to check if attendance for the date is found
            
            for attendance_doc in attendance_docs:
                # Check if the formatted date is in the document ID
                if formatted_date in attendance_doc.id:
                    found_attendance = True
                    attendance_data = attendance_doc.to_dict()
                    if attendance_data.get('status') == 'present':
                        present_students[pin_number] = student_data.get('name', 'N/A')
                    break  # No need to check further if attendance is found
            
            # If attendance document for the date is not found, mark as absent
            if not found_attendance:
                absent_students[pin_number] = student_data.get('name', 'N/A')
        
        return present_students, absent_students
    except Exception as e:
        print(f"Error fetching attendance: {e}")
        return {}, {}

# --------------------------- Handlers ---------------------------

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """
    Handles the /start and /help commands by sending a welcome message and options.
    """
    user_id = message.chat.id
    # Create a custom keyboard with options
    markup = types.ReplyKeyboardMarkup(row_width=2, one_time_keyboard=True)
    btn1 = types.KeyboardButton('/live_feed')
    btn2 = types.KeyboardButton('/attendance')
    markup.add(btn1, btn2)
    
    user_state[user_id] = None  # Reset user state
    bot.reply_to(message, "Welcome! Please choose an option:", reply_markup=markup)

@bot.message_handler(commands=['live_feed', 'attendance'])
def handle_option_selection(message):
    """
    Handles the selection of /live_feed and /attendance commands.
    """
    user_id = message.chat.id
    if message.text == '/live_feed':
        user_state[user_id] = 'live_feed'
        bot.reply_to(message, f"Here is your live feed link: {LIVE_FEED_LINK}")
    elif message.text == '/attendance':
        user_state[user_id] = 'awaiting_date'
        bot.reply_to(message, "Please enter the date for attendance in YYYY-MM-DD format:")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    """
    Handles all other messages, primarily for processing attendance dates.
    """
    user_id = message.chat.id
    state = user_state.get(user_id)
    
    if state == 'awaiting_date':
        date_input = message.text.strip()
        if validate_date_format(date_input):
            present_students, absent_students = fetch_attendance(date_input)
            
            if present_students:
                present_message = f"📚 *Present Students on {date_input}:*\n"
                present_message += "\n".join(f"{pin}: {name}" for pin, name in present_students.items())
            else:
                present_message = f"No present students found on {date_input}."
            
            bot.send_message(message.chat.id, present_message, parse_mode='Markdown')
            
            if absent_students:
                absent_message = f"❌ *Absent Students on {date_input}:*\n"
                absent_message += "\n".join(f"{pin}: {name}" for pin, name in absent_students.items())
            else:
                absent_message = "All students are present."
            
            bot.send_message(message.chat.id, absent_message, parse_mode='Markdown')
            
            user_state[user_id] = None
            
            markup = types.ReplyKeyboardRemove(selective=False)
            send_options(user_id)
        else:
            bot.reply_to(message, "❗ *Invalid date format.* Please enter the date in *YYYY-MM-DD* format:", parse_mode='Markdown')
    else:
        bot.reply_to(message, "ℹ️ Please choose an option from the keyboard or type /start to see options again.")


if __name__ == '__main__':
    print("Bot is running😁...")
    bot.infinity_polling()
