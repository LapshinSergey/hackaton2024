// ratingHandler.js

export function saveRating(userMessage, botMessage, rating, uniqueId) {
    const ratingData = {
        userMessage: userMessage,
        botMessage: botMessage,
        rating: rating
    };

    // Создание JSON файла (или отправка данных на сервер)
    const jsonString = JSON.stringify(ratingData);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const link = document.createElement('a');

    // Присваиваем уникальное имя файлу и скачиваем его
    link.href = URL.createObjectURL(blob);
    link.download = `rating_${uniqueId}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}
