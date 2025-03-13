"""
Telegram Bot for DeepSculpt.

This bot allows users to:
1. Generate 3D models through the DeepSculpt API
2. Configure generation parameters
3. View available models and options

To run the bot:
    python bot.py
"""

import os
import logging
import json
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ParseMode
from telegram.ext import (
    Updater,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    Filters,
    CallbackContext,
    ConversationHandler,
)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_TOKEN")
API_BASE_URL = os.environ.get("DEEPSCULPT_API_URL", "http://localhost:8000")

# Conversation states
SELECTING_MODEL, CONFIGURING_PARAMS, GENERATING = range(3)

# User session storage
user_sessions = {}


# Helper functions
def get_user_session(user_id: int) -> Dict[str, Any]:
    """Get or create user session."""
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "model_type": "skip",
            "num_samples": 1,
            "noise_dim": 100,
            "seed": None,
            "slice_axis": 0,
            "slice_position": 0.5,
            "last_request_id": None
        }
    return user_sessions[user_id]

def fetch_available_models() -> List[Dict[str, Any]]:
    """Fetch available models from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch models: {e}")
        return []

def generate_3d_model(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generate a 3D model using the API."""
    try:
        response = requests.post(f"{API_BASE_URL}/generate", json=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to generate model: {e}")
        return None


# Command handlers
async def start(update: Update, context: CallbackContext) -> int:
    """Handle the /start command."""
    user = update.effective_user
    message = (
        f"👋 Hello {user.first_name}!\n\n"
        f"Welcome to the DeepSculpt Telegram Bot. I can help you generate amazing 3D models "
        f"using deep learning.\n\n"
        f"Use /generate to start creating 3D models\n"
        f"Use /models to see available model types\n"
        f"Use /settings to configure generation parameters\n"
        f"Use /help to see all available commands"
    )
    await update.message.reply_text(message)
    return ConversationHandler.END

async def help_command(update: Update, context: CallbackContext) -> None:
    """Handle the /help command."""
    help_text = (
        "🤖 *DeepSculpt Bot Commands*\n\n"
        "/start - Start the bot\n"
        "/generate - Generate a new 3D model\n"
        "/models - Show available model types\n"
        "/settings - Configure generation parameters\n"
        "/status - Show current settings\n"
        "/help - Show this help message\n\n"
        "To generate a model, use /generate and follow the prompts."
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

async def show_models(update: Update, context: CallbackContext) -> None:
    """Handle the /models command."""
    models = fetch_available_models()
    
    if not models:
        await update.message.reply_text("❌ Failed to fetch available models. Please try again later.")
        return
    
    message = "🎮 *Available Model Types*\n\n"
    
    for model in models:
        message += f"*{model['model_type']}*: {model['description']}\n"
    
    message += "\nUse /generate to start creating with one of these models."
    
    await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

async def show_status(update: Update, context: CallbackContext) -> None:
    """Handle the /status command."""
    user_id = update.effective_user.id
    session = get_user_session(user_id)
    
    status_text = (
        "🔧 *Current Settings*\n\n"
        f"Model Type: `{session['model_type']}`\n"
        f"Number of Samples: `{session['num_samples']}`\n"
        f"Noise Dimension: `{session['noise_dim']}`\n"
        f"Random Seed: `{session['seed'] or 'Random'}`\n"
        f"Slice Axis: `{session['slice_axis']}`\n"
        f"Slice Position: `{session['slice_position']}`\n\n"
        "Use /settings to change these settings or /generate to create a model."
    )
    
    await update.message.reply_text(status_text, parse_mode=ParseMode.MARKDOWN)

async def start_generation(update: Update, context: CallbackContext) -> int:
    """Handle the /generate command."""
    models = fetch_available_models()
    
    if not models:
        await update.message.reply_text("❌ Failed to fetch available models. Please try again later.")
        return ConversationHandler.END
    
    keyboard = []
    for model in models:
        keyboard.append([
            InlineKeyboardButton(model["model_type"], callback_data=f"model:{model['model_type']}")
        ])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "🎮 Select a model type:", 
        reply_markup=reply_markup
    )
    
    return SELECTING_MODEL

async def model_selected(update: Update, context: CallbackContext) -> int:
    """Handle model selection callback."""
    query = update.callback_query
    await query.answer()
    
    model_type = query.data.split(":")[1]
    user_id = update.effective_user.id
    session = get_user_session(user_id)
    session["model_type"] = model_type
    
    # Sample count selection
    keyboard = [
        [
            InlineKeyboardButton("1", callback_data="samples:1"),
            InlineKeyboardButton("4", callback_data="samples:4"),
            InlineKeyboardButton("9", callback_data="samples:9"),
            InlineKeyboardButton("16", callback_data="samples:16")
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        f"Model type: *{model_type}*\n\n"
        f"How many samples do you want to generate?",
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )
    
    return CONFIGURING_PARAMS

async def samples_selected(update: Update, context: CallbackContext) -> int:
    """Handle sample count selection callback."""
    query = update.callback_query
    await query.answer()
    
    num_samples = int(query.data.split(":")[1])
    user_id = update.effective_user.id
    session = get_user_session(user_id)
    session["num_samples"] = num_samples
    
    # Slice axis selection
    keyboard = [
        [
            InlineKeyboardButton("X Axis", callback_data="axis:0"),
            InlineKeyboardButton("Y Axis", callback_data="axis:1"),
            InlineKeyboardButton("Z Axis", callback_data="axis:2")
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        f"Model type: *{session['model_type']}*\n"
        f"Samples: *{num_samples}*\n\n"
        f"Select the slice axis for visualization:",
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )
    
    return CONFIGURING_PARAMS

async def axis_selected(update: Update, context: CallbackContext) -> int:
    """Handle slice axis selection callback."""
    query = update.callback_query
    await query.answer()
    
    slice_axis = int(query.data.split(":")[1])
    user_id = update.effective_user.id
    session = get_user_session(user_id)
    session["slice_axis"] = slice_axis
    
    # Position selection
    keyboard = [
        [
            InlineKeyboardButton("0.25", callback_data="position:0.25"),
            InlineKeyboardButton("0.5", callback_data="position:0.5"),
            InlineKeyboardButton("0.75", callback_data="position:0.75")
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        f"Model type: *{session['model_type']}*\n"
        f"Samples: *{session['num_samples']}*\n"
        f"Slice axis: *{['X', 'Y', 'Z'][slice_axis]}*\n\n"
        f"Select the slice position (0.0-1.0):",
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )
    
    return CONFIGURING_PARAMS

async def position_selected(update: Update, context: CallbackContext) -> int:
    """Handle slice position selection callback."""
    query = update.callback_query
    await query.answer()
    
    slice_position = float(query.data.split(":")[1])
    user_id = update.effective_user.id
    session = get_user_session(user_id)
    session["slice_position"] = slice_position
    
    # Final confirmation
    keyboard = [
        [
            InlineKeyboardButton("🔄 Generate", callback_data="generate"),
            InlineKeyboardButton("❌ Cancel", callback_data="cancel")
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        f"🔧 *Generation Settings*\n\n"
        f"Model type: *{session['model_type']}*\n"
        f"Samples: *{session['num_samples']}*\n"
        f"Slice axis: *{['X', 'Y', 'Z'][session['slice_axis']]}*\n"
        f"Slice position: *{session['slice_position']}*\n\n"
        f"Ready to generate?",
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )
    
    return GENERATING

async def generate_model(update: Update, context: CallbackContext) -> int:
    """Handle generate button click."""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    session = get_user_session(user_id)
    
    await query.edit_message_text("🔄 Generating your 3D model... Please wait.")
    
    # Prepare parameters for API request
    params = {
        "model_type": session["model_type"],
        "num_samples": session["num_samples"],
        "noise_dim": session["noise_dim"],
        "seed": session["seed"],
        "slice_axis": session["slice_axis"],
        "slice_position": session["slice_position"]
    }
    
    # Generate the model
    result = generate_3d_model(params)
    
    if result is None:
        await context.bot.send_message(
            chat_id=user_id,
            text="❌ Failed to generate model. Please try again later."
        )
        return ConversationHandler.END
    
    # Store the request ID
    session["last_request_id"] = result["request_id"]
    
    # Get the image URL
    image_url = f"{API_BASE_URL}{result['image_url']}"
    
    # Send the image
    caption = (
        f"🎮 *DeepSculpt Generation*\n\n"
        f"Model: *{result['model_type']}*\n"
        f"Samples: *{params['num_samples']}*\n"
        f"Generation time: *{result['generation_time']:.2f}s*\n"
        f"Request ID: `{result['request_id']}`\n\n"
        f"Use /generate to create another model."
    )
    
    try:
        # Download and send the image
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        
        await context.bot.send_photo(
            chat_id=user_id,
            photo=image_response.content,
            caption=caption,
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception as e:
        logger.error(f"Failed to send image: {e}")
        await context.bot.send_message(
            chat_id=user_id,
            text=f"✅ Model generated successfully, but failed to send image.\n\nYou can view it at: {image_url}"
        )
    
    return ConversationHandler.END

async def cancel_generation(update: Update, context: CallbackContext) -> int:
    """Handle cancel button click."""
    query = update.callback_query
    await query.answer()
    
    await query.edit_message_text("❌ Generation cancelled.")
    return ConversationHandler.END

async def settings_command(update: Update, context: CallbackContext) -> None:
    """Handle the /settings command."""
    user_id = update.effective_user.id
    session = get_user_session(user_id)
    
    # Create inline keyboard for settings
    keyboard = [
        [InlineKeyboardButton("Model Type", callback_data="setting:model_type")],
        [InlineKeyboardButton("Number of Samples", callback_data="setting:num_samples")],
        [InlineKeyboardButton("Random Seed", callback_data="setting:seed")],
        [InlineKeyboardButton("Advanced Settings", callback_data="setting:advanced")]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "⚙️ *Settings*\n\n"
        "Select a setting to change:",
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )

async def setting_selected(update: Update, context: CallbackContext) -> None:
    """Handle settings selection."""
    query = update.callback_query
    await query.answer()
    
    setting = query.data.split(":")[1]
    user_id = update.effective_user.id
    session = get_user_session(user_id)
    
    if setting == "model_type":
        models = fetch_available_models()
        
        if not models:
            await query.edit_message_text("❌ Failed to fetch available models. Please try again later.")
            return
        
        keyboard = []
        for model in models:
            keyboard.append([
                InlineKeyboardButton(model["model_type"], callback_data=f"set_model:{model['model_type']}")
            ])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "🎮 Select a model type:", 
            reply_markup=reply_markup
        )
    
    elif setting == "num_samples":
        keyboard = [
            [
                InlineKeyboardButton("1", callback_data="set_samples:1"),
                InlineKeyboardButton("4", callback_data="set_samples:4"),
                InlineKeyboardButton("9", callback_data="set_samples:9"),
                InlineKeyboardButton("16", callback_data="set_samples:16")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "🔢 Select number of samples:", 
            reply_markup=reply_markup
        )
    
    elif setting == "seed":
        # Options for seed
        keyboard = [
            [
                InlineKeyboardButton("Random", callback_data="set_seed:null"),
                InlineKeyboardButton("1234", callback_data="set_seed:1234"),
                InlineKeyboardButton("5678", callback_data="set_seed:5678")
            ],
            [
                InlineKeyboardButton("Enter custom seed", callback_data="set_seed:custom")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "🎲 Select a random seed:", 
            reply_markup=reply_markup
        )
    
    elif setting == "advanced":
        # Advanced settings
        keyboard = [
            [InlineKeyboardButton("Slice Axis", callback_data="setting:slice_axis")],
            [InlineKeyboardButton("Slice Position", callback_data="setting:slice_position")],
            [InlineKeyboardButton("Noise Dimension", callback_data="setting:noise_dim")],
            [InlineKeyboardButton("Back to Main Settings", callback_data="setting:back")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "⚙️ *Advanced Settings*\n\n"
            "Select a setting to change:",
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )

async def update_setting(update: Update, context: CallbackContext) -> None:
    """Handle setting updates."""
    query = update.callback_query
    await query.answer()
    
    parts = query.data.split(":")
    setting_type = parts[0]
    value = parts[1]
    
    user_id = update.effective_user.id
    session = get_user_session(user_id)
    
    if setting_type == "set_model":
        session["model_type"] = value
        message = f"✅ Model type set to: *{value}*"
    
    elif setting_type == "set_samples":
        session["num_samples"] = int(value)
        message = f"✅ Number of samples set to: *{value}*"
    
    elif setting_type == "set_seed":
        if value == "null":
            session["seed"] = None
            message = "✅ Using random seed for each generation"
        elif value == "custom":
            # Prompt for custom seed
            await query.edit_message_text(
                "Please enter a custom seed (integer number):"
            )
            # Store that we're waiting for a custom seed
            context.user_data["awaiting_custom_seed"] = True
            return
        else:
            session["seed"] = int(value)
            message = f"✅ Random seed set to: *{value}*"
    
    elif setting_type == "set_axis":
        session["slice_axis"] = int(value)
        message = f"✅ Slice axis set to: *{['X', 'Y', 'Z'][int(value)]}*"
    
    elif setting_type == "set_position":
        session["slice_position"] = float(value)
        message = f"✅ Slice position set to: *{value}*"
    
    elif setting_type == "set_noise_dim":
        session["noise_dim"] = int(value)
        message = f"✅ Noise dimension set to: *{value}*"
    
    else:
        message = "❌ Unknown setting type"
    
    await query.edit_message_text(
        f"{message}\n\nUse /settings to change more settings or /generate to create a model.",
        parse_mode=ParseMode.MARKDOWN
    )

async def handle_custom_seed(update: Update, context: CallbackContext) -> None:
    """Handle custom seed input."""
    # Check if we're waiting for a custom seed
    if not context.user_data.get("awaiting_custom_seed"):
        return
    
    user_id = update.effective_user.id
    session = get_user_session(user_id)
    
    try:
        seed = int(update.message.text)
        session["seed"] = seed
        await update.message.reply_text(
            f"✅ Random seed set to: *{seed}*\n\n"
            f"Use /settings to change more settings or /generate to create a model.",
            parse_mode=ParseMode
        )
    except ValueError:
        await update.message.reply_text(
            "❌ Invalid seed. Please enter an integer number."
        )
    
    # Reset the awaiting flag
    context.user_data["awaiting_custom_seed"] = False


def main() -> None:
    """Start the bot."""
    # Create the Updater and pass it your bot's token
    updater = Updater(TELEGRAM_TOKEN)
    
    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher
    
    # Set up conversation handler for generation
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("generate", start_generation)],
        states={
            SELECTING_MODEL: [
                CallbackQueryHandler(model_selected, pattern=r'^model:')
            ],
            CONFIGURING_PARAMS: [
                CallbackQueryHandler(samples_selected, pattern=r'^samples:'),
                CallbackQueryHandler(axis_selected, pattern=r'^axis:'),
                CallbackQueryHandler(position_selected, pattern=r'^position:')
            ],
            GENERATING: [
                CallbackQueryHandler(generate_model, pattern=r'^generate$'),
                CallbackQueryHandler(cancel_generation, pattern=r'^cancel$')
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel_generation)],
    )
    
    # Add conversation handler
    dispatcher.add_handler(conv_handler)
    
    # Add command handlers
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("models", show_models))
    dispatcher.add_handler(CommandHandler("status", show_status))
    dispatcher.add_handler(CommandHandler("settings", settings_command))
    
    # Add callback query handlers
    dispatcher.add_handler(CallbackQueryHandler(setting_selected, pattern=r'^setting:'))
    dispatcher.add_handler(CallbackQueryHandler(update_setting, pattern=r'^set_'))
    
    # Add message handler for custom seed
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_custom_seed))
    
    # Start the Bot
    updater.start_polling()
    
    # Run the bot until the user presses Ctrl-C
    updater.idle()


if __name__ == "__main__":
    print("Starting DeepSculpt Telegram Bot...")
    main()