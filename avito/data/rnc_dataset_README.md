Синтетический датасет для задачи по дополнительным микрокатегориям в Ремонте и отделке.

Файлы:
- rnc_dataset.csv
- rnc_dataset.jsonl

Что внутри:
- 3000 объявлений
- 11 микрокатегорий из исходного справочника
- кейсы разных типов:
  * single_direct
  * turnkey_no_split
  * turnkey_split
  * multi_service_split
  * multi_service_no_split
  * bullets_mixed
  * noisy_short

Колонки:
- itemId: идентификатор объявления
- sourceMcId: исходная микрокатегория объявления
- sourceMcTitle: название исходной микрокатегории
- description: текст объявления
- targetDetectedMcIds: список всех найденных mcId в тексте
- targetSplitMcIds: список mcId, по которым стоит создать дополнительные draft
- shouldSplit: есть ли хотя бы один дополнительный draft
- caseType: тип сценария
- split: train / val / test

Статистика:
- всего записей: 3000
- доля shouldSplit=true: 0.371
- train/val/test: {'train': 2094, 'test': 459, 'val': 447}
- распределение кейсов: {'single_direct': 843, 'turnkey_no_split': 544, 'turnkey_split': 535, 'bullets_mixed': 307, 'multi_service_no_split': 294, 'multi_service_split': 252, 'noisy_short': 225}
