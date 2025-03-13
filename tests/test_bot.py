"""
Tests for the DeepSculpt Telegram bot.
"""

import pytest
import os
import sys
import json
from unittest.mock import patch, MagicMock, AsyncMock

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import bot module
import bot


@pytest.fixture
def setup_env():
    """Setup environment variables for tests."""
    # Save original environment
    original_env = dict(os.environ)
    
    # Set test environment variables
    os.environ["TELEGRAM_BOT_TOKEN"] = "test_token"
    os.environ["DEEPSCULPT_API_URL"] = "http://localhost:8000"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_update():
    """Create a mock Update object."""
    mock = MagicMock()
    mock.effective_user = MagicMock()
    mock.effective_user.id = 12345
    mock.effective_user.first_name = "Test User"
    mock.message = MagicMock()
    mock.message.reply_text = AsyncMock()
    
    return mock


@pytest.fixture
def mock_context():
    """Create a mock CallbackContext object."""
    mock = MagicMock()
    mock.bot = MagicMock()
    mock.bot.send_message = AsyncMock()
    mock.bot.send_photo = AsyncMock()
    mock.user_data = {}
    
    return mock


@pytest.fixture
def mock_query():
    """Create a mock callback query."""
    mock = MagicMock()
    mock.answer = AsyncMock()
    mock.edit_message_text = AsyncMock()
    mock.data = "test_data"
    mock.message = MagicMock()
    
    return mock


@patch("bot.requests.get")
@patch("bot.fetch_available_models")
async def test_start_command(mock_fetch_models, mock_requests_get, mock_update, mock_context, setup_env):
    """Test the start command handler."""
    # Call the handler
    result = await bot.start(mock_update, mock_context)
    
    # Check that reply_text was called
    mock_update.message.reply_text.assert_called_once()
    assert "Welcome to the DeepSculpt Telegram Bot" in mock_update.message.reply_text.call_args[0][0]
    
    # Check return value
    assert result == -1  # ConversationHandler.END


@patch("bot.requests.get")
@patch("bot.fetch_available_models")
async def test_help_command(mock_fetch_models, mock_requests_get, mock_update, mock_context, setup_env):
    """Test the help command handler."""
    # Call the handler
    await bot.help_command(mock_update, mock_context)
    
    # Check that reply_text was called with help text
    mock_update.message.reply_text.assert_called_once()
    assert "DeepSculpt Bot Commands" in mock_update.message.reply_text.call_args[0][0]


@patch("bot.fetch_available_models")
async def test_show_models(mock_fetch_models, mock_update, mock_context, setup_env):
    """Test the show_models command handler."""
    # Setup mock
    mock_fetch_models.return_value = [
        {"model_type": "simple", "description": "Test model 1"},
        {"model_type": "complex", "description": "Test model 2"}
    ]
    
    # Call the handler
    await bot.show_models(mock_update, mock_context)
    
    # Check that reply_text was called with models info
    mock_update.message.reply_text.assert_called_once()
    
    call_args = mock_update.message.reply_text.call_args[0][0]
    assert "Available Model Types" in call_args
    assert "simple" in call_args
    assert "complex" in call_args


async def test_show_status(mock_update, mock_context):
    """Test the show_status command handler."""
    # Set up a user session
    user_id = mock_update.effective_user.id
    bot.user_sessions[user_id] = {
        "model_type": "skip",
        "num_samples": 4,
        "noise_dim": 100,
        "seed": None,
        "slice_axis": 0,
        "slice_position": 0.5
    }
    
    # Call the handler
    await bot.show_status(mock_update, mock_context)
    
    # Check that reply_text was called with status info
    mock_update.message.reply_text.assert_called_once()
    
    call_args = mock_update.message.reply_text.call_args[0][0]
    assert "Current Settings" in call_args
    assert "skip" in call_args
    assert "4" in call_args


@patch("bot.fetch_available_models")
async def test_start_generation(mock_fetch_models, mock_update, mock_context, setup_env):
    """Test the start_generation command handler."""
    # Setup mock
    mock_fetch_models.return_value = [
        {"model_type": "simple", "description": "Test model 1"},
        {"model_type": "complex", "description": "Test model 2"}
    ]
    
    # Call the handler
    result = await bot.start_generation(mock_update, mock_context)
    
    # Check that reply_text was called with model selection
    mock_update.message.reply_text.assert_called_once()
    assert "Select a model type" in mock_update.message.reply_text.call_args[0][0]
    
    # Check return value
    assert result == 0  # SELECTING_MODEL


@patch("bot.get_user_session")
async def test_model_selected(mock_get_session, mock_query, mock_context):
    """Test the model_selected callback handler."""
    # Setup mocks
    mock_query.data = "model:skip"
    mock_session = {
        "model_type": "simple",
        "num_samples": 1,
        "slice_axis": 0,
        "slice_position": 0.5
    }
    mock_get_session.return_value = mock_session
    
    # Call the handler
    result = await bot.model_selected(mock_query, mock_context)
    
    # Check that query.answer and edit_message_text were called
    mock_query.answer.assert_called_once()
    mock_query.edit_message_text.assert_called_once()
    
    # Check that session was updated
    assert mock_session["model_type"] == "skip"
    
    # Check that message contains sample selection
    call_args = mock_query.edit_message_text.call_args[0][0]
    assert "How many samples" in call_args
    
    # Check return value
    assert result == 1  # CONFIGURING_PARAMS


@patch("bot.get_user_session")
async def test_samples_selected(mock_get_session, mock_query, mock_context):
    """Test the samples_selected callback handler."""
    # Setup mocks
    mock_query.data = "samples:4"
    mock_session = {
        "model_type": "skip",
        "num_samples": 1,
        "slice_axis": 0,
        "slice_position": 0.5
    }
    mock_get_session.return_value = mock_session
    
    # Call the handler
    result = await bot.samples_selected(mock_query, mock_context)
    
    # Check that query.answer and edit_message_text were called
    mock_query.answer.assert_called_once()
    mock_query.edit_message_text.assert_called_once()
    
    # Check that session was updated
    assert mock_session["num_samples"] == 4
    
    # Check that message contains axis selection
    call_args = mock_query.edit_message_text.call_args[0][0]
    assert "Select the slice axis" in call_args
    
    # Check return value
    assert result == 1  # CONFIGURING_PARAMS


@patch("bot.get_user_session")
async def test_axis_selected(mock_get_session, mock_query, mock_context):
    """Test the axis_selected callback handler."""
    # Setup mocks
    mock_query.data = "axis:1"
    mock_session = {
        "model_type": "skip",
        "num_samples": 4,
        "slice_axis": 0,
        "slice_position": 0.5
    }
    mock_get_session.return_value = mock_session
    
    # Call the handler
    result = await bot.axis_selected(mock_query, mock_context)
    
    # Check that query.answer and edit_message_text were called
    mock_query.answer.assert_called_once()
    mock_query.edit_message_text.assert_called_once()
    
    # Check that session was updated
    assert mock_session["slice_axis"] == 1
    
    # Check that message contains position selection
    call_args = mock_query.edit_message_text.call_args[0][0]
    assert "Select the slice position" in call_args
    
    # Check return value
    assert result == 1  # CONFIGURING_PARAMS


@patch("bot.get_user_session")
async def test_position_selected(mock_get_session, mock_query, mock_context):
    """Test the position_selected callback handler."""
    # Setup mocks
    mock_query.data = "position:0.75"
    mock_session = {
        "model_type": "skip",
        "num_samples": 4,
        "slice_axis": 1,
        "slice_position": 0.5
    }
    mock_get_session.return_value = mock_session
    
    # Call the handler
    result = await bot.position_selected(mock_query, mock_context)
    
    # Check that query.answer and edit_message_text were called
    mock_query.answer.assert_called_once()
    mock_query.edit_message_text.assert_called_once()
    
    # Check that session was updated
    assert mock_session["slice_position"] == 0.75
    
    # Check that message contains confirmation
    call_args = mock_query.edit_message_text.call_args[0][0]
    assert "Generation Settings" in call_args
    assert "Ready to generate" in call_args
    
    # Check return value
    assert result == 2  # GENERATING


@patch("bot.get_user_session")
@patch("bot.generate_3d_model")
async def test_generate_model(mock_generate, mock_get_session, mock_query, mock_context, setup_env):
    """Test the generate_model callback handler."""
    # Setup mocks
    mock_query.data = "generate"
    user_id = 12345
    mock_query.message.chat_id = user_id
    
    mock_session = {
        "model_type": "skip",
        "num_samples": 4,
        "noise_dim": 100,
        "seed": None,
        "slice_axis": 1,
        "slice_position": 0.75
    }
    mock_get_session.return_value = mock_session
    
    mock_result = {
        "request_id": "test123",
        "image_url": "/outputs/test.png",
        "model_type": "skip",
        "generation_time": 1.5,
        "parameters": {
            "num_samples": 4,
            "noise_dim": 100,
            "seed": None,
            "slice_axis": 1,
            "slice_position": 0.75
        }
    }
    mock_generate.return_value = mock_result
    
    # Mock requests.get
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.content = b"test_image_data"
    
    with patch("bot.requests.get", return_value=mock_response):
        # Call the handler
        result = await bot.generate_model(mock_query, mock_context)
    
    # Check that query.answer and edit_message_text were called
    mock_query.answer.assert_called_once()
    mock_query.edit_message_text.assert_called_once()
    
    # Check that generate_3d_model was called with correct parameters
    mock_generate.assert_called_once_with({
        "model_type": "skip",
        "num_samples": 4,
        "noise_dim": 100,
        "seed": None,
        "slice_axis": 1,
        "slice_position": 0.75
    })
    
    # Check that send_photo was called
    mock_context.bot.send_photo.assert_called_once()
    
    # Check return value
    assert result == -1  # ConversationHandler.END


@patch("bot.get_user_session")
async def test_cancel_generation(mock_get_session, mock_query, mock_context):
    """Test the cancel_generation callback handler."""
    # Setup mocks
    mock_query.data = "cancel"
    
    # Call the handler
    result = await bot.cancel_generation(mock_query, mock_context)
    
    # Check that query.answer and edit_message_text were called
    mock_query.answer.assert_called_once()
    mock_query.edit_message_text.assert_called_once()
    
    # Check message content
    call_args = mock_query.edit_message_text.call_args[0][0]
    assert "Generation cancelled" in call_args
    
    # Check return value
    assert result == -1  # ConversationHandler.END


@patch("bot.get_user_session")
async def test_settings_command(mock_get_session, mock_update, mock_context):
    """Test the settings_command handler."""
    # Setup mock
    mock_session = {
        "model_type": "skip",
        "num_samples": 4,
        "noise_dim": 100,
        "seed": None,
        "slice_axis": 1,
        "slice_position": 0.75
    }
    mock_get_session.return_value = mock_session
    
    # Call the handler
    await bot.settings_command(mock_update, mock_context)
    
    # Check that reply_text was called
    mock_update.message.reply_text.assert_called_once()
    
    # Check message content
    call_args = mock_update.message.reply_text.call_args[0][0]
    assert "Settings" in call_args


@patch("bot.get_user_session")
@patch("bot.fetch_available_models")
async def test_setting_selected(mock_fetch_models, mock_get_session, mock_query, mock_context, setup_env):
    """Test the setting_selected callback handler."""
    # Setup mocks
    mock_query.data = "setting:model_type"
    mock_session = {
        "model_type": "skip",
        "num_samples": 4,
        "noise_dim": 100,
        "seed": None,
        "slice_axis": 1,
        "slice_position": 0.75
    }
    mock_get_session.return_value = mock_session
    
    mock_fetch_models.return_value = [
        {"model_type": "simple", "description": "Test model 1"},
        {"model_type": "complex", "description": "Test model 2"}
    ]
    
    # Call the handler
    await bot.setting_selected(mock_query, mock_context)
    
    # Check that query.answer and edit_message_text were called
    mock_query.answer.assert_called_once()
    mock_query.edit_message_text.assert_called_once()
    
    # Check message content
    call_args = mock_query.edit_message_text.call_args[0][0]
    assert "Select a model type" in call_args


@patch("bot.get_user_session")
async def test_update_setting(mock_get_session, mock_query, mock_context):
    """Test the update_setting callback handler."""
    # Setup mocks
    mock_query.data = "set_model:complex"
    mock_session = {
        "model_type": "skip",
        "num_samples": 4,
        "noise_dim": 100,
        "seed": None,
        "slice_axis": 1,
        "slice_position": 0.75
    }
    mock_get_session.return_value = mock_session
    
    # Call the handler
    await bot.update_setting(mock_query, mock_context)
    
    # Check that query.answer and edit_message_text were called
    mock_query.answer.assert_called_once()
    mock_query.edit_message_text.assert_called_once()
    
    # Check that session was updated
    assert mock_session["model_type"] == "complex"
    
    # Check message content
    call_args = mock_query.edit_message_text.call_args[0][0]
    assert "Model type set to" in call_args


async def test_handle_custom_seed(mock_update, mock_context):
    """Test the handle_custom_seed handler."""
    # Setup mocks
    mock_context.user_data["awaiting_custom_seed"] = True
    mock_update.message.text = "12345"
    
    user_id = mock_update.effective_user.id
    bot.user_sessions[user_id] = {
        "model_type": "skip",
        "num_samples": 4,
        "noise_dim": 100,
        "seed": None,
        "slice_axis": 1,
        "slice_position": 0.75
    }
    
    # Call the handler
    await bot.handle_custom_seed(mock_update, mock_context)
    
    # Check that session was updated
    assert bot.user_sessions[user_id]["seed"] == 12345
    
    # Check that reply_text was called
    mock_update.message.reply_text.assert_called_once()
    
    # Check that awaiting flag was reset
    assert not mock_context.user_data["awaiting_custom_seed"]


@patch("requests.get")
def test_fetch_available_models(mock_get, setup_env):
    """Test the fetch_available_models function."""
    # Setup mock
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = [
        {"model_type": "simple", "description": "Test model 1"},
        {"model_type": "complex", "description": "Test model 2"}
    ]
    mock_get.return_value = mock_response
    
    # Call the function
    models = bot.fetch_available_models()
    
    # Check return value
    assert len(models) == 2
    assert models[0]["model_type"] == "simple"
    assert models[1]["model_type"] == "complex"
    
    # Check that get was called with correct URL
    mock_get.assert_called_once_with(f"{os.environ['DEEPSCULPT_API_URL']}/models")


@patch("requests.post")
def test_generate_3d_model(mock_post, setup_env):
    """Test the generate_3d_model function."""
    # Setup mock
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "request_id": "test123",
        "image_url": "/outputs/test.png",
        "model_type": "skip",
        "generation_time": 1.5,
        "parameters": {
            "num_samples": 4,
            "noise_dim": 100,
            "seed": None,
            "slice_axis": 1,
            "slice_position": 0.75
        }
    }
    mock_post.return_value = mock_response
    
    # Call the function
    params = {
        "model_type": "skip",
        "num_samples": 4,
        "noise_dim": 100,
        "seed": None,
        "slice_axis": 1,
        "slice_position": 0.75
    }
    result = bot.generate_3d_model(params)
    
    # Check return value
    assert result["request_id"] == "test123"
    assert result["image_url"] == "/outputs/test.png"
    
    # Check that post was called with correct URL and data
    mock_post.assert_called_once_with(
        f"{os.environ['DEEPSCULPT_API_URL']}/generate",
        json=params
    )


def test_get_user_session():
    """Test the get_user_session function."""
    # Clear user_sessions
    bot.user_sessions.clear()
    
    # Call the function for a new user
    user_id = 12345
    session = bot.get_user_session(user_id)
    
    # Check that a new session was created
    assert user_id in bot.user_sessions
    assert session["model_type"] == "skip"
    assert session["num_samples"] == 1
    assert session["noise_dim"] == 100
    assert session["seed"] is None
    
    # Call the function again for the same user
    session["model_type"] = "complex"
    new_session = bot.get_user_session(user_id)
    
    # Check that the same session was returned
    assert new_session["model_type"] == "complex"