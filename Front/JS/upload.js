// Элемент для выбора файла
const fileInput = document.getElementById('file-input');

// Обработчик для загрузки файла на сервер после выбора
fileInput.addEventListener('change', async (event) => {
    const file = event.target.files[0];

    if (!file) {
        showModalMessage("Пожалуйста, выберите файл для загрузки.");
        return;
    }

    showLoadingModal(); // Показать модальное окно с анимацией загрузки

    try {
        await uploadFile(file);
        updateLoadingModalMessage("Файл успешно загружен на сервер!", true);
    } catch (error) {
        console.error("Ошибка при загрузке файла:", error);
        updateLoadingModalMessage("Ошибка при загрузке файла на сервер.", true);
    }
});

// Функция для загрузки файла на сервер
async function uploadFile(file) {
    const url = 'http://100.88.62.188:8000/upload';
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(url, {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        throw new Error(`Ошибка сети: ${response.statusText}`);
    }

    return response.json();
}

// Функция для показа модального окна загрузки
function showLoadingModal() {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    overlay.id = 'loading-modal';

    const modalContent = document.createElement('div');
    modalContent.className = 'modal-content';

    const loader = document.createElement('div');
    loader.className = 'loader-ring';

    const message = document.createElement('p');
    message.textContent = "Загрузка файла...";
    message.id = 'loading-message';

    modalContent.appendChild(loader);
    modalContent.appendChild(message);
    overlay.appendChild(modalContent);
    document.body.appendChild(overlay);
}

// Функция для обновления сообщения в модальном окне
function updateLoadingModalMessage(newMessage, showButton = false) {
    const loader = document.querySelector('.loader-ring');
    if (loader) {
        loader.style.display = 'none'; // Скрываем кольцо загрузки
    }

    const message = document.getElementById('loading-message');
    if (message) {
        message.textContent = newMessage; // Обновляем текст
    }

    // Добавляем кнопку "Ок", только если showButton = true
    if (showButton) {
        const closeButton = document.createElement('button');
        closeButton.textContent = "Ок";
        closeButton.addEventListener('click', hideLoadingModal);
        message.parentNode.appendChild(closeButton);
    }
}

// Функция для скрытия модального окна загрузки
function hideLoadingModal() {
    const overlay = document.getElementById('loading-modal');
    if (overlay) {
        overlay.remove(); // Удаляем модальное окно из DOM
    }
}

// Функция для показа модального окна с сообщением
function showModalMessage(messageText) {
    showLoadingModal(); // Используем уже существующее модальное окно
    updateLoadingModalMessage(messageText, true);
}

// CSS для модального окна
const style = document.createElement('style');
style.textContent = `
    .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        margin-top: 100px
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
    }
    
    .modal-content {
        background: #fff;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        max-width: 300px;
        width: 100%;
    }
    
    .loader-ring {
        margin-bottom: 15px;
        /* Добавьте сюда стили анимации кольца, если необходимо */
    }
    
    button {
        margin-top: 15px;
        padding: 8px 16px;
        border: none;
        background-color: #007bff;
        color: white;
        border-radius: 4px;
        cursor: pointer;
    }
    
    button:hover {
        background-color: #0056b3;
    }
`;
document.head.appendChild(style);
