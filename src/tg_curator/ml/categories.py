"""The built-in categories a topic can be made from (DESIGN §10, ``research/models.md`` §9.4).

Twenty keys with English labels and nine short, Telegram-like seed phrases each (three in
English, Russian and Uzbek Latin). The seeds are the only training data for crypto, ads and
humor, which no public news dataset covers, so they are data, not an afterthought: a translator
can extend them and the prototypes are recomputed from them. The order of ``CATEGORIES`` is the
order of the shipped model's output columns and must not change without retraining the head.
"""

from __future__ import annotations

import difflib

from tg_curator.domain import Category
from tg_curator.errors import ConfigError

# key -> (English label, what goes in it)
CATEGORIES: dict[str, tuple[str, str]] = {
    "tech": ("Technology & AI", "software, gadgets, AI models, IT companies, apps, internet"),
    "science": ("Science & space", "research, space, physics, biology, climate, discoveries"),
    "finance": ("Finance & economy", "markets, central banks, rates, inflation, banking, deals"),
    "crypto": ("Crypto", "bitcoin, ethereum, exchanges, tokens, DeFi, stablecoins"),
    "politics": ("Politics & government", "elections, parliament, laws, decrees, ministries"),
    "world": ("World & conflict", "international news, wars, UN, sanctions, protests"),
    "sport": ("Sport", "football, tennis, UFC, Olympics, matches, transfers, results"),
    "health": ("Health & medicine", "diseases, vaccines, hospitals, medicine, mental health"),
    "education": ("Education", "schools, universities, exams, admissions, students"),
    "culture": ("Culture & entertainment", "cinema, music, books, TV, celebrities, festivals"),
    "real_estate": ("Real estate", "housing prices, mortgages, apartments, construction, rent"),
    "jobs": ("Jobs & career", "vacancies, hiring, salaries, labour market"),
    "travel": ("Travel & tourism", "visas, flights, airlines, hotels, tourism"),
    "auto": ("Auto & transport", "cars, EVs, recalls, public transport, roads"),
    "society": ("Society & crime", "crime, police, courts, fraud, accidents, social issues"),
    "disaster": ("Weather & disasters", "earthquakes, floods, fires, storms, forecasts"),
    "religion": ("Religion", "religions, holidays, religious leaders"),
    "lifestyle": ("Lifestyle", "food, fitness, sleep, fashion, relationships, home, tips"),
    "humor": ("Humor & memes", "jokes, memes, funny observations"),
    "ads": ("Ads & promotions", "discounts, giveaways, sponsored posts, referral links"),
}

KEYS: tuple[str, ...] = tuple(CATEGORIES)

# Categories the model confuses most (measured): when the best guess is one of a pair the
# second guess is worth showing. ``jobs`` and ``society`` are the weak ones.
NEIGHBOUR_PAIRS: tuple[tuple[str, str], ...] = (
    ("finance", "crypto"),
    ("tech", "science"),
    ("politics", "world"),
    ("society", "disaster"),
    ("jobs", "tech"),
)

SEEDS: dict[str, tuple[str, ...]] = {
    "tech": (
        "New AI model released with longer context and lower API prices",
        "Apple unveils a new iPhone with a faster chip",
        "Startup launches an app; the update adds new features",
        "Вышла новая модель ИИ с большим контекстом и дешёвым API",
        "Apple представила новый iPhone с новым чипом",
        "Стартап выпустил приложение, обновление добавило функции",
        "Yangi sun’iy intellekt modeli chiqdi, API narxi arzonlashdi",
        "Apple yangi chipli iPhone’ni taqdim etdi",
        "Startap yangi ilova chiqardi, yangilanishda yangi funksiyalar",
    ),
    "science": (
        "Astronomers discover an exoplanet with water in its atmosphere",
        "SpaceX rocket launch and landing succeeded",
        "Scientists publish a study in Nature on a new material",
        "Астрономы открыли экзопланету с водой в атмосфере",
        "Ракета SpaceX успешно стартовала и приземлилась",
        "Учёные опубликовали в Nature исследование нового материала",
        "Astronomlar atmosferasida suv bor ekzosayyorani topdi",
        "SpaceX raketasi muvaffaqiyatli uchdi va qo‘ndi",
        "Olimlar Nature jurnalida yangi material haqida tadqiqot chop etdi",
    ),
    "finance": (
        "The central bank changed its key interest rate; inflation forecast",
        "Company reports quarterly revenue and profit, shares move",
        "Dollar exchange rate and stock market today",
        "Центробанк изменил ключевую ставку, прогноз по инфляции",
        "Компания отчиталась о квартальной выручке и прибыли, акции выросли",
        "Курс доллара и фондовый рынок сегодня",
        "Markaziy bank asosiy stavkani o‘zgartirdi, inflyatsiya prognozi",
        "Kompaniya choraklik daromad va foyda haqida hisobot berdi",
        "Dollar kursi va fond bozori bugun",
    ),
    "crypto": (
        "Bitcoin price hits a new high; ETF inflows and liquidations",
        "Crypto exchange hacked, millions in tokens stolen",
        "Ethereum and altcoins fall; stablecoin regulation",
        "Биткоин обновил максимум; притоки в ETF и ликвидации",
        "Криптобиржу взломали, украдены миллионы в токенах",
        "Эфир и альткоины падают; регулирование стейблкоинов",
        "Bitkoin narxi yangi rekordga chiqdi; ETF oqimlari",
        "Kripto birja buzildi, millionlab token o‘g‘irlandi",
        "Efir va altkoinlar tushdi; steyblkoinlar tartibga solinadi",
    ),
    "politics": (
        "The president signed a decree; parliament passed a law",
        "Election results and the new government coalition",
        "Minister announced a reform of the tax code",
        "Президент подписал указ; парламент принял закон",
        "Итоги выборов и новая правительственная коалиция",
        "Министр объявил о реформе налогового кодекса",
        "Prezident farmon imzoladi; parlament qonun qabul qildi",
        "Saylov natijalari va yangi hukumat koalitsiyasi",
        "Vazir soliq kodeksi islohotini e’lon qildi",
    ),
    "world": (
        "UN Security Council resolution on the ceasefire; war and peace talks",
        "Sanctions, missile strikes and the front line",
        "Protests and clashes in the capital; international reaction",
        "Резолюция Совбеза ООН о прекращении огня; война и переговоры",
        "Санкции, ракетные удары и линия фронта",
        "Протесты и столкновения в столице; реакция мирового сообщества",
        "BMT Xavfsizlik Kengashi o‘t ochishni to‘xtatish rezolyutsiyasi; urush va muzokaralar",
        "Sanksiyalar, raketa zarbalari va front chizig‘i",
        "Poytaxtda norozilik namoyishlari va to‘qnashuvlar",
    ),
    "sport": (
        "Full time: the match ended 2-1, goals and the league table",
        "Tennis final result; Grand Slam title",
        "UFC fighter defends the title; boxing and the Olympics",
        "Матч закончился со счётом 2:1, голы и турнирная таблица",
        "Результат теннисного финала; титул Большого шлема",
        "Боец UFC защитил титул; бокс и Олимпиада",
        "O‘yin 2:1 hisobida yakunlandi, gollar va turnir jadvali",
        "Tennis finali natijasi; Katta dulduk unvoni",
        "UFC jangchisi unvonini himoya qildi; boks va Olimpiada",
    ),
    "health": (
        "WHO recommends a vaccine; flu season and hospitals",
        "Doctors warn about a disease outbreak; treatment and medicines",
        "Mental health, diabetes, heart disease study",
        "ВОЗ рекомендует вакцину; сезон гриппа и больницы",
        "Врачи предупреждают о вспышке болезни; лечение и лекарства",
        "Психическое здоровье, диабет, исследование болезней сердца",
        "JSST vaksinani tavsiya qildi; gripp mavsumi va shifoxonalar",
        "Shifokorlar kasallik tarqalishi haqida ogohlantirdi; davolash va dorilar",
        "Ruhiy salomatlik, diabet, yurak kasalliklari tadqiqoti",
    ),
    "education": (
        "University admission exam results published; students and schools",
        "New school year, curriculum and teachers",
        "Scholarships and state grants for students",
        "Опубликованы результаты вступительных экзаменов; студенты и школы",
        "Новый учебный год, программа и учителя",
        "Стипендии и государственные гранты для студентов",
        "Kirish imtihonlari natijalari e’lon qilindi; talabalar va maktablar",
        "Yangi o‘quv yili, dastur va o‘qituvchilar",
        "Talabalar uchun stipendiya va davlat grantlari",
    ),
    "culture": (
        "Film wins the award at the festival; new album and concert tour",
        "TV series premiere, actors and the box office",
        "Book, exhibition and theatre news; celebrities",
        "Фильм получил награду фестиваля; новый альбом и концертный тур",
        "Премьера сериала, актёры и кассовые сборы",
        "Книги, выставки и театр; знаменитости",
        "Film festivalda mukofot oldi; yangi albom va konsert",
        "Serial premyerasi, aktyorlar va kassa",
        "Kitob, ko‘rgazma va teatr yangiliklari; mashhurlar",
    ),
    "real_estate": (
        "Apartment prices rose; price per square metre and mortgages",
        "New residential complex and construction; rent",
        "Housing market, developers and mortgage rates",
        "Цены на квартиры выросли; стоимость квадратного метра и ипотека",
        "Новый жилой комплекс и стройка; аренда",
        "Рынок жилья, застройщики и ставки по ипотеке",
        "Kvartira narxlari oshdi; kvadrat metr narxi va ipoteka",
        "Yangi turar-joy majmuasi va qurilish; ijara",
        "Uy-joy bozori, quruvchilar va ipoteka stavkalari",
    ),
    "jobs": (
        "Vacancy: developer wanted, salary range, send your CV",
        "Hiring and layoffs; labour market and wages",
        "Job offer: remote work, requirements and benefits",
        "Вакансия: требуется разработчик, вилка зарплаты, резюме",
        "Найм и сокращения; рынок труда и зарплаты",
        "Работа: удалённо, требования и условия",
        "Vakansiya: dasturchi kerak, maosh, rezyume yuboring",
        "Ishga olish va qisqartirishlar; mehnat bozori va maoshlar",
        "Ish taklifi: masofaviy, talablar va shartlar",
    ),
    "travel": (
        "Visa-free regime introduced; new flights and airlines",
        "Tourism season, hotels and destinations",
        "Airport opened; travel tips and tickets",
        "Введён безвизовый режим; новые рейсы и авиакомпании",
        "Туристический сезон, отели и направления",
        "Открыт аэропорт; советы путешественникам и билеты",
        "Vizasiz rejim joriy etildi; yangi reyslar va aviakompaniyalar",
        "Turizm mavsumi, mehmonxonalar va yo‘nalishlar",
        "Aeroport ochildi; sayohat maslahatlari va chiptalar",
    ),
    "auto": (
        "Car maker launches an electric model; plant and production",
        "Vehicle recall over a defect; software update",
        "New car prices, taxis and public transport",
        "Автопроизводитель выпустил электромобиль; завод и производство",
        "Отзыв автомобилей из-за дефекта; обновление ПО",
        "Цены на новые машины, такси и общественный транспорт",
        "Avtomobil ishlab chiqaruvchi elektromobil chiqardi; zavod",
        "Nuqson tufayli avtomobillar chaqirib olindi",
        "Yangi mashinalar narxi, taksi va jamoat transporti",
    ),
    "society": (
        "Police detained a fraud group; court and sentence",
        "Road accident in the city; victims and investigation",
        "Social issues, families, migrants and communities",
        "Полиция задержала группу мошенников; суд и приговор",
        "ДТП в городе; пострадавшие и расследование",
        "Социальные проблемы, семьи, мигранты и общество",
        "Politsiya firibgarlar guruhini ushladi; sud va hukm",
        "Shaharda yo‘l-transport hodisasi; jabrlanganlar va tergov",
        "Ijtimoiy muammolar, oilalar, migrantlar va jamiyat",
    ),
    "disaster": (
        "Earthquake of magnitude 6; tsunami warning and casualties",
        "Heat wave and storm warning from forecasters",
        "Flood and wildfire; emergency services evacuate residents",
        "Землетрясение магнитудой 6; угроза цунами и жертвы",
        "Жара и штормовое предупреждение синоптиков",
        "Наводнение и лесной пожар; спасатели эвакуируют жителей",
        "6 magnitudali zilzila; sunami xavfi va qurbonlar",
        "Jazirama va sinoptiklarning bo‘ron ogohlantirishi",
        "Suv toshqini va o‘rmon yong‘ini; qutqaruvchilar aholini evakuatsiya qildi",
    ),
    "religion": (
        "Ramadan start date announced; mosque and prayer",
        "Church holiday, the Pope and believers",
        "Religious leaders, pilgrimage and Hajj",
        "Объявлена дата начала Рамадана; мечеть и молитва",
        "Церковный праздник, Папа Римский и верующие",
        "Религиозные лидеры, паломничество и хадж",
        "Ramazon boshlanish sanasi e’lon qilindi; masjid va namoz",
        "Cherkov bayrami, Rim papasi va dindorlar",
        "Diniy yetakchilar, ziyorat va haj",
    ),
    "lifestyle": (
        "Five habits for better sleep and fitness",
        "Recipe of the day and restaurant tips",
        "Fashion, relationships and home advice",
        "Пять привычек для хорошего сна и формы",
        "Рецепт дня и советы по ресторанам",
        "Мода, отношения и советы по дому",
        "Yaxshi uyqu va fitnes uchun beshta odat",
        "Kun retsepti va restoran maslahatlari",
        "Moda, munosabatlar va uy maslahatlari",
    ),
    "humor": (
        "Me: I'll go to bed early. Also me at 3am 😂",
        "Monday mood meme, funny joke",
        "Joke about programmers and coffee",
        "Я: лягу пораньше. Тоже я в 3 ночи 😂",
        "Мем про понедельник, смешной анекдот",
        "Анекдот про программистов и кофе",
        "Men: erta yotaman. Yana men tunda soat 3 da 😂",
        "Dushanba kayfiyati mem, kulgili hazil",
        "Dasturchilar va qahva haqida hazil",
    ),
    "ads": (
        "Discount 60% only until Sunday, sign up by the link",
        "Giveaway: subscribe to the sponsors and win a prize",
        "Our VPN works everywhere, 3 days free, advertisement",
        "Скидка 60% только до воскресенья, записывайся по ссылке",
        "Розыгрыш: подпишись на спонсоров и выиграй приз",
        "Наш VPN работает везде, 3 дня бесплатно, реклама",
        "60% chegirma faqat yakshanbagacha, havola orqali yoziling",
        "Sovg‘a o‘yini: homiylarga obuna bo‘ling va yutib oling",
        "Bizning VPN hamma joyda ishlaydi, 3 kun bepul, reklama",
    ),
}


def all() -> list[Category]:  # noqa: A001 - the name is fixed by the contract (§8)
    """The built-in categories in their stable order."""
    return [Category(key=key, label=label) for key, (label, _) in CATEGORIES.items()]


def label(key: str) -> str:
    """The English label of a category key (``KeyError`` for an unknown key)."""
    return CATEGORIES[key][0]


def describe(key: str) -> str:
    """One line saying what goes into the category, for ``/topics add`` and the CLI list."""
    return CATEGORIES[key][1]


def suggest(key: str) -> list[str]:
    """Up to two category keys close to an unknown ``key``: ``"technology"`` -> ``["tech"]``.

    Matched against the keys and the words of the English labels (``"Technology & AI"``), so
    the words the spec and the labels use lead to the key that stands for them.
    """
    wanted = key.strip().casefold()
    if not wanted:
        return []
    names: dict[str, str] = {k: k for k in CATEGORIES}
    for k, (lbl, _) in CATEGORIES.items():
        for word in "".join(ch if ch.isalnum() else " " for ch in lbl.casefold()).split():
            if len(word) > 1:
                names.setdefault(word, k)
    out: list[str] = []
    for k in CATEGORIES:  # a prefix either way: "tech" in "technology", "sport" in "sports"
        if len(wanted) >= 3 and (k.startswith(wanted) or wanted.startswith(k)):
            out.append(k)
    if wanted in names and names[wanted] not in out:
        out.insert(0, names[wanted])
    if not out:
        for name in difflib.get_close_matches(wanted, list(names), n=5, cutoff=0.75):
            if names[name] not in out:
                out.append(names[name])
    return out[:2]


def unknown_key_message(key: str) -> str:
    """The sentence for a category key that is not built in, naming close matches."""
    close = suggest(key)
    hint = ""
    if close:
        hint = " (did you mean " + " or ".join(f"'{k}'" for k in close) + "?)"
    return f"unknown category '{key}'{hint}; `curator topics categories` lists the keys"


def require_key(key: str) -> str:
    """The built-in key ``key`` names (case and surrounding blanks ignored); ``ConfigError``
    naming close matches and where the list is when there is none."""
    wanted = key.strip().casefold()
    if wanted in CATEGORIES:
        return wanted
    raise ConfigError(unknown_key_message(key.strip()))


def seed_texts() -> tuple[list[str], list[str]]:
    """Every seed phrase with its category key, in ``CATEGORIES`` order."""
    texts: list[str] = []
    keys: list[str] = []
    for key in KEYS:
        for seed in SEEDS[key]:
            texts.append(seed)
            keys.append(key)
    return texts, keys
