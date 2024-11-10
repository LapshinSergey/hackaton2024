const messagesContainer = document.getElementById('chat-messages');
const sendButton = document.getElementById('send-button');
const wsJson = new WebSocket("ws://100.88.62.188:8000/ask"); // WebSocket для запросов JSON-файлов 
const inputField = document.getElementById('chat-input');



// Обработка нажатия Enter для отправки сообщения
inputField.addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault();
        sendMessage();
    }
});

wsJson.onmessage = (event) => {
     userMessage = inputField.value; // Получаем текущее сообщение пользователя
    
    try {
        const parsedData = JSON.parse(event.data);

        // Проверка типа сообщения
        if (parsedData.type === "chat") {
            displayBotMessage(parsedData, userMessage);
        } else {
            console.log("Получено сообщение не для чата, обработка пропущена.");
        }
    } catch (error) {
        console.error("Ошибка при обработке данных от сервера:", error);
    }

    sendButton.classList.remove('loading');
    sendButton.disabled = false;
};

// Функция для отправки сообщения
function sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (message === "") return;

    displayUserMessage(message);
    input.value = '';

    sendButton.classList.add('loading');
    sendButton.disabled = true;

    // Проверяем готово ли соединение WebSocket, и отправляем сообщение
    if (wsJson.readyState === WebSocket.OPEN) {
        wsJson.send(message);
    } else if (wsJson.readyState === WebSocket.CONNECTING) {
        wsJson.addEventListener('open', function () {
            wsJson.send(message);
        }, { once: true });
    } else {
        console.error("WebSocket соединение не готово");
        sendButton.classList.remove('loading');
        sendButton.disabled = false;
    }
}

// Функция для отображения сообщения пользователя
function displayUserMessage(message) {
    const userMessage = document.createElement('div');
    userMessage.className = 'message user-message';
    userMessage.textContent = message;
    messagesContainer.appendChild(userMessage);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}






// Функция для отображения сообщения бота и добавления рейтинга
function displayBotMessage(data, userMessage) {
    const uniqueId = Date.now();
    const botMessage = document.createElement('div');
    botMessage.className = 'message bot-message';

    let parsedData;
    let totalRelevanceCount = 0;

    try {
        parsedData = JSON.parse(data);
        if (!parsedData || Object.keys(parsedData).length === 0) {
            parsedData = { answer: "Нет данных", relevance: [] };
        }
    } catch (e) {
        console.error("Ошибка при парсинге JSON:", e);
        parsedData = { answer: "Ошибка при обработке данных", relevance: [] };
    }

    const botMessageContent = parsedData.answer || "Нет данных";
    botMessage.innerHTML = `<p>${botMessageContent}</p>`;
    

    if (parsedData.relevance && Array.isArray(parsedData.relevance)) {
        totalRelevanceCount = parsedData.relevance.length;
    }

    let tableHTML = `
        <table class="table table-no-border">
            <thead>
                <tr>
                    <th>File</th>
                    <th>Pages</th>
                    <th>Score</th>
                    <th>Squeeze</th>
                </tr>
            </thead>
            <tbody>
    `;

    if (parsedData.relevance && parsedData.relevance.length > 0) {
        tableHTML += parsedData.relevance.slice(0, 1).map(row => `
            <tr>
                <td>${row.file || 'Нет данных'}</td>
                <td>${(row.pages || []).join(', ') || 'Нет данных'}</td>
                <td>${row.score ? row.score.toFixed(2) : 'Нет данных'}</td>
                <td>${row.squeeze || 'Нет данных'}</td>
            </tr>
        `).join('');
    } else {
        tableHTML += `
            <tr>
                <td colspan="4" style="text-align: center;">Данные не найдены</td>
            </tr>
        `;
    }
    tableHTML += `</tbody></table>`;
    botMessage.innerHTML += tableHTML;

    const starRatingContainer = document.createElement('div');
    starRatingContainer.className = 'rating-container';

    const relevanceCountElem = document.createElement('p');
    relevanceCountElem.textContent = `Найдено релевантных ответов: ${totalRelevanceCount} `;

    const detailsButton = document.createElement('button');
    detailsButton.textContent = '(cмотреть все)';
    detailsButton.classList.add('details-button');
    detailsButton.addEventListener('click', () => openModal(parsedData.relevance, uniqueId));

    relevanceCountElem.appendChild(detailsButton);
    starRatingContainer.appendChild(relevanceCountElem);

    // Добавляем контейнер для звездного рейтинга
    const starRating = document.createElement('div');
    starRating.className = 'star-rating';

    for (let i = 5; i >= 1; i--) {
        const starInput = document.createElement('input');
        starInput.type = 'radio';
        starInput.name = `rating-${uniqueId}`;
        starInput.id = `rating-${uniqueId}-${i}`;
        starInput.value = i;

        const starLabel = document.createElement('label');
        starLabel.htmlFor = `rating-${uniqueId}-${i}`;
        starLabel.className = 'star';
        starLabel.textContent = '★';

        // Обработчик на изменение значения (выбор звезды)
        starInput.addEventListener('change', () => {
            const ratingData = {
                question: userMessage, // Сообщение пользователя
                answer: botMessageContent, // Ответ от бота
                score: i // Выбранный рейтинг (число звёзд)
            };
            console.log("Отправляемый JSON:", JSON.stringify(ratingData));


            // Отправляем JSON с рейтингом на сервер через POST запрос
            fetch('http://100.88.62.188:8000/vote', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(ratingData)
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Ошибка сети при отправке рейтинга');
                }
                return response.json();
            })
            .then(data => {
                console.log("Рейтинг успешно отправлен:", data);
            })
            .catch(error => {
                console.error("Ошибка при отправке рейтинга:", error);
            });
        });

        // Добавляем звезды в контейнер
        starRating.appendChild(starInput);
        starRating.appendChild(starLabel);
    }

    starRatingContainer.appendChild(starRating);
    botMessage.appendChild(starRatingContainer);
    messagesContainer.appendChild(botMessage);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}























// Функция для открытия модального окна с релевантными ответами
function openModal(relevanceData, uniqueId) {
    // Создаём контейнер модального окна
    const modalContainer = document.createElement('div');
    modalContainer.classList.add('modal-container');
    modalContainer.id = `modal-${uniqueId}`;

    // Создаём содержимое модального окна
    const modalContent = `
        <div class="modal-content">
            <span class="close-button" onclick="closeModal('${uniqueId}')">&times;</span>
            <h2>Релевантные ответы</h2>
            <table class="table">
                <thead>
                    <tr>
                        <th>File</th>
                        <th>Pages</th>
                        <th>Score</th>
                        <th>Squeeze</th>
                    </tr>
                </thead>
                <tbody>
                    ${relevanceData.map(row => `
                        <tr>
                            <td>${row.file || 'Нет данных'}</td>
                            <td>${(row.pages || []).join(', ') || 'Нет данных'}</td>
                            <td>${row.score ? row.score.toFixed(2) : 'Нет данных'}</td>
                            <td>${row.squeeze || 'Нет данных'}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;

    modalContainer.innerHTML = modalContent;

    // Добавляем модальное окно в документ
    document.body.appendChild(modalContainer);

    // Отображаем модальное окно
    modalContainer.style.display = 'block';
}

// Функция для закрытия модального окна
function closeModal(uniqueId) {
    const modal = document.getElementById(`modal-${uniqueId}`);
    if (modal) {
        modal.style.display = 'none';
        modal.remove(); // Удаляем элемент из DOM после закрытия
    }
}

// Обработка входящих сообщений от WebSocket-сервера
wsJson.onmessage = (event) => {
    displayBotMessage(event.data);  // Сообщение от WebSocket для JSON-файлов
    sendButton.classList.remove('loading');
    sendButton.disabled = false;
};

// Слушатель на кнопку отправки сообщения
sendButton.addEventListener('click', sendMessage);








