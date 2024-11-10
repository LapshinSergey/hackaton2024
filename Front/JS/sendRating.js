// sendRating.js
function sendRating(question, answer, score) {
    // Формируем JSON с рейтингом
    const ratingData = {
        question: question, // Сообщение пользователя
        answer: answer,     // Ответ от бота
        score: score        // Рейтинг в звездах
    };

    // Отправляем JSON на сервер
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
}
