# Поддержка Linux для FramePack-eichi (неофициальная)

Этот каталог содержит неофициальные скрипты поддержки для использования FramePack-eichi в среде Linux. Эти скрипты предоставляются для удобства и **не являются официально поддерживаемыми**. Используйте их на свой страх и риск.

## Системные требования

- **ОС**: Рекомендуется Ubuntu 22.04 LTS или новее (также могут работать другие дистрибутивы с поддержкой Python 3.10)
- **CPU**: Рекомендуется современный многоядерный CPU с 8 или более ядрами
- **RAM**: Минимум 16ГБ, рекомендуется 32ГБ или больше (64ГБ рекомендуется для сложной обработки или высокого разрешения)
- **GPU**: NVIDIA RTX 30XX/40XX/50XX серии (8ГБ VRAM или больше)
- **VRAM**: Минимум 8ГБ (рекомендуется 12ГБ или больше)
- **Хранилище**: 150ГБ или больше свободного места (рекомендуется SSD)
- **Необходимое программное обеспечение**:
  - CUDA Toolkit 12.6
  - Python 3.10.x
  - PyTorch 2.6 с поддержкой CUDA

## Включенные скрипты

- `update.sh` - Скрипт для обновления из основного репозитория и применения изменений FramePack-eichi
- `setup_submodule.sh` - Скрипт для первоначальной настройки
- `install_linux.sh` - Простой установщик для Linux
- `run_endframe_ichi.sh` - Скрипт запуска обычной версии на японском
- `run_endframe_ichi_ru.sh` - Скрипт запуска обычной версии на русском
- `run_endframe_ichi_f1.sh` - Скрипт запуска версии F1 на японском
- `run_endframe_ichi_f1_ru.sh` - Скрипт запуска версии F1 на русском
- `run_oneframe_ichi.sh` - Скрипт запуска версии вывода одного кадра на японском
- `run_oneframe_ichi_ru.sh` - Скрипт запуска версии вывода одного кадра на русском
- Скрипты запуска для других языков

## Процедура настройки Linux (подмодуль)

### 1. Установка предварительных требований

```bash
# Обновление системных пакетов
sudo apt update && sudo apt upgrade -y

# Установка базовых инструментов разработки и библиотек
sudo apt install -y git wget ffmpeg libavformat-dev libavdevice-dev libavfilter-dev libswscale-dev libopenblas-dev

# Установка CUDA Toolkit 12.6
# Примечание: Следуйте официальным инструкциям NVIDIA для установки CUDA Toolkit
# https://developer.nvidia.com/cuda-12-6-0-download-archive

# Установка Python 3.10
sudo apt install -y python3.10 python3.10-venv python3-pip
```

### 2. Клонирование и настройка FramePack-eichi

```bash
# Клонирование репозитория FramePack-eichi
git clone https://github.com/git-ai-code/FramePack-eichi.git
cd FramePack-eichi

# Создание и активация виртуальной среды
python3.10 -m venv venv
source venv/bin/activate

# Настройка подмодуля (автоматически загружает оригинальный FramePack)
./linux/setup_submodule.sh

# Установка PyTorch с поддержкой CUDA и зависимостей
cd webui/submodules/FramePack
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### 3. Запуск FramePack-eichi

```bash
# Возврат в корневой каталог FramePack-eichi
cd ~/FramePack-eichi  # Настройте путь в соответствии с вашим местом установки

# Использование скриптов запуска
./linux/run_endframe_ichi.sh       # Стандартная версия/японский UI
./linux/run_endframe_ichi_f1.sh    # Версия с моделью F1/японский UI
./linux/run_oneframe_ichi.sh       # Версия вывода одного кадра/японский UI
./linux/run_endframe_ichi_ru.sh    # Стандартная версия/русский UI
./linux/run_endframe_ichi_f1_ru.sh # Версия с моделью F1/русский UI
./linux/run_oneframe_ichi_ru.sh    # Версия вывода одного кадра/русский UI

# Другие языковые версии
./linux/run_endframe_ichi_en.sh    # Английский UI
./linux/run_endframe_ichi_zh-tw.sh # UI на традиционном китайском
```

## Использование

### Настройка существующего репозитория

```bash
cd /path/to/FramePack-eichi
./linux/setup_submodule.sh
```

### Отражение обновлений из основного проекта

```bash
cd /path/to/FramePack-eichi
./linux/update.sh
```

### Запуск приложения

```bash
cd /path/to/FramePack-eichi
./linux/run_endframe_ichi.sh     # Обычная версия/японский
./linux/run_endframe_ichi_f1.sh  # Версия F1/японский
./linux/run_oneframe_ichi.sh     # Версия вывода одного кадра/японский
./linux/run_endframe_ichi_ru.sh  # Обычная версия/русский
./linux/run_endframe_ichi_f1_ru.sh # Версия F1/русский
./linux/run_oneframe_ichi_ru.sh  # Версия вывода одного кадра/русский
```

## Установка библиотек ускорения

Если при запуске FramePack отображаются следующие сообщения, библиотеки ускорения не установлены:

```
Xformers is not installed!
Flash Attn is not installed!
Sage Attn is not installed!
```

Установка этих библиотек повысит скорость обработки (ожидается ускорение примерно на 30%).

### Метод установки

Выполните следующие команды в соответствии с вашей средой Python:

```bash
# 1. Перейдите в директорию FramePack
cd /path/to/FramePack-eichi/webui/submodules/FramePack

# 2. Установите необходимые библиотеки
pip install xformers triton
pip install packaging ninja
pip install flash-attn --no-build-isolation
pip install sage-attn==1.0.6

# 3. Перезапустите для проверки установки
```

### Установка библиотек ускорения в автономной среде

Для автономной установки используйте следующий метод:

```bash
# Убедитесь, что виртуальная среда активирована
source venv/bin/activate

# Перейдите в директорию FramePack
cd FramePack

# Установите библиотеки ускорения
pip install xformers triton
pip install packaging ninja
pip install flash-attn --no-build-isolation 
pip install sageattention==1.0.6
```

### Примечания по установке

- Поддерживается только при использовании CUDA 12.x (для CUDA 11.x некоторые библиотеки необходимо собирать)
- Установка `flash-attn` может быть трудной в некоторых средах. В таком случае, улучшение производительности ожидается даже с одним Xformers
- Убедитесь, что версия PyTorch 2.0.0 или выше
- Пакет sage-attn может быть переименован в sageattention (указывайте версию 1.0.6)

## Устранение неполадок

### Ошибка «CUDA out of memory»

Если возникает нехватка памяти, попробуйте следующие меры:

1. Закройте другие приложения, использующие GPU
2. Уменьшите размер изображения (рекомендуется около 512x512)
3. Уменьшите размер батча
4. Увеличьте значение `gpu_memory_preservation` (более высокое значение уменьшает использование памяти, но снижает скорость обработки)

### Проблемы с установкой CUDA и совместимостью

Если отображается ошибка «CUDA недоступна» или предупреждение «Переключение на выполнение CPU»:

1. Проверьте, правильно ли установлен CUDA:
   ```bash
   nvidia-smi
   ```

2. Проверьте, распознаёт ли PyTorch CUDA:
   ```python
   python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
   ```

3. Убедитесь, что вы используете поддерживаемый GPU (рекомендуется RTX 30XX, 40XX или 50XX серии)

4. Проверьте совместимость драйвера CUDA и PyTorch:
   - Драйвер, совместимый с CUDA 12.6
   - PyTorch 2.6 с поддержкой CUDA

### Сбой загрузки модели

Если возникают ошибки при загрузке сегментов модели:

1. Убедитесь, что модели правильно загружены
2. При первом запуске дождитесь автоматической загрузки необходимых моделей (около 30ГБ)
3. Убедитесь, что у вас достаточно места на диске (рекомендуется минимум 150ГБ)

## Предостережения

- Эти скрипты не имеют официальной поддержки
- Если возникают ошибки, связанные с путями выполнения, пожалуйста, измените скрипты соответствующим образом
- Использование памяти увеличивается при сложной обработке или высоком разрешении (рекомендуется достаточный RAM и GPU с высоким VRAM)
- Если после длительного использования возникает утечка памяти, перезапустите приложение
- Вы можете регистрировать вопросы или отчеты об ошибках как Issues, но решение не гарантируется

## Справочная информация

- Официальный FramePack: https://github.com/lllyasviel/FramePack
- FramePack-eichi: https://github.com/git-ai-code/FramePack-eichi
- Установка библиотек ускорения: https://github.com/lllyasviel/FramePack/issues/138
- CUDA Toolkit: https://developer.nvidia.com/cuda-12-6-0-download-archive