#!/bin/bash
set -e
echo "=== 1. Установка библиотек ==="
pip install -r requirements.txt

echo "=== 2. Сборка модулей pyparadigm ==="
if [ ! -d "Sigmorphon2018SharedTask" ]; then
    git clone https://github.com/AlexeySorokin/Sigmorphon2018SharedTask.git
fi
cd Sigmorphon2018SharedTask
python3 setup_aligner.py build_ext --inplace
cd pyparadigm
python3 setup.py build_ext --inplace
cd ../..

echo "=== 3. Загрузка обучающих данных из Ossetic-COT ==="
if [ ! -d "data" ]; then
    git clone https://github.com/ania3000/Ossetic-COT.git data
fi

echo "=== Готово! ==="
