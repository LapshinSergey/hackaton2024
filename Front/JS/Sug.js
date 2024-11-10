const suggestionsContainer = document.getElementById('suggestions');
const websocket1 = new WebSocket('ws://100.88.62.188:8000/prompt');

websocket1.onopen = () => {
    console.log("WebSocket подключен.");
};

// Обработка сообщения от сервера
websocket1.onmessage = function(event) {
    const filteredSuggestions = JSON.parse(event.data);
    showFilteredSuggestions(filteredSuggestions);
};

// Функция для отображения предложений
function showFilteredSuggestions(filteredSuggestions) {
    suggestionsContainer.innerHTML = ''; // Очищаем предыдущие подсказки
    if (filteredSuggestions.length > 0) {
        filteredSuggestions.forEach(suggestion => {
            const suggestionItem = document.createElement('div');
            suggestionItem.className = 'suggestion-item';
            suggestionItem.textContent = suggestion;
            suggestionItem.onclick = () => {
                document.getElementById('chat-input').value = suggestion; // Устанавливаем значение в поле ввода
                suggestionsContainer.innerHTML = ''; // Очищаем подсказки после выбора
            };
            suggestionsContainer.appendChild(suggestionItem); // Добавляем элемент подсказки в контейнер
        });
    }
}

// Обработка нажатия клавиш
document.getElementById('chat-input').addEventListener('input', function(event) {
    const input = event.target.value;
    if (input) {
        websocket1.send(input); // Отправляем ввод на сервер для автозаполнения
    } else {
        suggestionsContainer.innerHTML = ''; // Очищаем, если ввод пуст
    }
});

function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

// Отправка сообщения
function sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (message === "") return;

    displayUserMessage(message); // Отображаем сообщение пользователя
    input.value = ''; // Очищаем поле ввода
}

// Отображение сообщения пользователя
function displayUserMessage(message) {
    const userMessage = document.createElement('div');
    userMessage.className = 'message user-message';
    userMessage.textContent = message;
    document.getElementById('chat-messages').appendChild(userMessage);
    document.getElementById('chat-messages').scrollTop = document.getElementById('chat-messages').scrollHeight; // Прокручиваем вниз
}
