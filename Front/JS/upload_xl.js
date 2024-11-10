// Проверяем наличие элемента для выбора xlsx файла
document.addEventListener("DOMContentLoaded", () => {
    const xlsxFileInput = document.getElementById('xlsx-file-input');

    if (!xlsxFileInput) {
        console.error("Элемент с ID 'xlsx-file-input' не найден.");
        return;
    }

    // Обработчик для загрузки только xlsx файла на сервер после выбора
    xlsxFileInput.addEventListener('change', async (event) => {
        const file = event.target.files[0];
        
        if (!file) {
            showModalMessage("Пожалуйста, выберите файл.");
            return;
        }

        if (!file.name.endsWith('.xlsx')) {
            showModalMessage("Пожалуйста, выберите файл в формате .xlsx.");
            return;
        }

        showLoadingModal(); // Показать модальное окно с анимацией загрузки

        try {
            const responseText = await uploadXlsxFile(file);
            updateLoadingModalMessage("Файл .xlsx успешно загружен на сервер!");
            console.log("Ответ сервера:", responseText); // Проверяем ответ сервера
        } catch (error) {
            console.error("Ошибка при загрузке файла .xlsx:", error);
            updateLoadingModalMessage("Ошибка при загрузке файла на сервер.");
        } finally {
            setTimeout(hideLoadingModal, 2000); // Закрываем модальное окно через 2 секунды
        }
    });
});

// Функция для загрузки xlsx файла на другой сервер
async function uploadXlsxFile(file) {
    const url = 'http://100.88.62.188:8000/process_questions';
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(url, {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        throw new Error(`Ошибка сети: ${response.statusText}`);
    }

    // Читаем ответ как текст, а не JSON, чтобы избежать ошибки парсинга
    return response.text();
}
