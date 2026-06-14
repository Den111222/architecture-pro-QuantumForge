#!/bin/bash
# Настройка ежедневного обновления индекса

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/kb_update.log"

# Проверка прав
if [ "$EUID" -ne 0 ]; then
    echo "Запустите с sudo: sudo ./cron_setup.sh"
    exit 1
fi

# Добавление задания в crontab
CRON_JOB="0 6 * * * cd $SCRIPT_DIR && /usr/bin/python3 update_index.py >> $LOG_FILE 2>&1"

# Проверка, существует ли уже задание
if crontab -l 2>/dev/null | grep -q "update_index.py"; then
    echo "Задание уже существует в crontab"
else
    (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
    echo "Добавлено задание: $CRON_JOB"
fi

# Создание лог-файла
touch $LOG_FILE
chmod 644 $LOG_FILE

echo "Логи будут писаться в $LOG_FILE"
echo "Проверить статус: tail -f $LOG_FILE"