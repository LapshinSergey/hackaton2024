const websocket = new WebSocket('ws://localhost:8765');

// Когда WebSocket подключен
websocket.onopen = () => {
    console.log("WebSocket подключен.");
};

// Обработка сообщений от сервера
websocket.onmessage = function(event) {
    const message = event.data;  // Получаем сообщение от сервера
    displayBotMessage(message);  // Отображаем это сообщение как ответ бота
};

// Отправка сообщения через WebSocket
function sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (message === "") return;

    displayUserMessage(message);  // Отображаем сообщение пользователя в чате
    websocket.send(message);      // Отправляем сообщение на сервер через WebSocket
    input.value = '';             // Очищаем поле ввода
}

// Функция для отображения сообщений от пользователя
function displayUserMessage(message) {
    const userMessage = document.createElement('div');
    userMessage.className = 'message user-message';
    userMessage.textContent = message;
    document.getElementById('chat-messages').appendChild(userMessage);
    document.getElementById('chat-messages').scrollTop = document.getElementById('chat-messages').scrollHeight;
}

// Функция для отображения сообщений от бота (или от сервера)
function displayBotMessage(message) {
    const botMessage = document.createElement('div');
    botMessage.className = 'message bot-message';
    botMessage.textContent = message;
    document.getElementById('chat-messages').appendChild(botMessage);
    document.getElementById('chat-messages').scrollTop = document.getElementById('chat-messages').scrollHeight;
}
