#!/bin/bash
set -e

echo "=== 1. Установка библиотек ==="
pip install -r requirements.txt

echo "=== 2. Сборка C++ модулей pyparadigm ==="
if [ ! -d "Sigmorphon2018SharedTask" ]; then
    git clone https://github.com/AlexeySorokin/Sigmorphon2018SharedTask.git
fi

# 2.1 Компиляция aligner (заходим внутрь папки aligner)
echo "Компиляция aligner..."
cd Sigmorphon2018SharedTask/aligner
python3 setup_aligner.py build_ext --inplace
cd ../..

# 2.2 Компиляция pyparadigm
echo "Компиляция pyparadigm..."
cd Sigmorphon2018SharedTask/pyparadigm
python3 setup.py build_ext --inplace
cd ../..

echo "=== 3. Загрузка обучающих данных из Ossetic-COT ==="
if [ ! -d "data" ]; then
    git clone https://github.com/ania3000/Ossetic-COT.git data
fi

echo "=== Готово! Все модули успешно скомпилированы. ==="
