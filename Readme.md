Данный проект создан в рамках конкурса "Лидеры цифровой трансформации 2024". Решаемая задача: Предиктивная модель для рекомендации продуктов банка.
Для решения задачи были предоставлены следующие данные:
● Эмбеддинги диалогов со службами технической поддержки;
● Транзакционная активность;
● Гео-информация о местах постоянного или временного пребывания в виде кодов.

Описание модулей:
● lct_training_trx.py - модуль обработки последовательностей транзакционной активности с использованием библиотеки pytorch-lifestream;
● lct_training_text.py - модуль обработки эмбеддингов диалогов. В качестве первичной обработки данных использована библиотека pytorch-lifestream;
● lct_training_geo.py - модуль обработки гео-информации. В модуле использован подход обучения эмбеддингов схожести клиентов на основе данных о их местоположении. Подход реализован с использованием библиотеки pytorch.  
● lct_training_history.py - использование исторических данных поведения клиентов, создание целевых меток класса. В качестве первичной обработки данных использована библиотека pytorch-lifestream;
● lct_training_trx_features.py - модуль построения дополнительных параметров на основе группировки данных о транзакциях клиентов за текущий месяц;
● catboost_training.py - основной модуль сбора подготовленных данных, дополнительной обработки и обучения моделей. Обучение выполнялось с использованием библиотеки catboost. Итоговое обучение проводилось на нескольких фолдах со спосом разбиения StrtifiedKFold с использованием нескольких сидов. Результаты ансамбля усреднялись.

В рамках проекта, кроме обозначенных выше, использовались стандартные классы и методы из библиотек pandas, numpy, sklearn др. 