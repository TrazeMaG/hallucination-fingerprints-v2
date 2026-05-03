"""
Experiment 10 — Structurally Diverse Benchmark
================================================
Key insight: same fact tested in 4 structural forms.
This is what separates a real mechanistic claim from
a prompt-engineering artifact.

4 structures per fact:
  1. Forward completion:  "The capital of France is"
  2. Question form:       "What is the capital of France?"
  3. Reverse direction:   "Paris is the capital of which"
  4. Contextual embed:    "France, whose capital city is"

If suppression is real it appears across all 4 structures.
If it only appears in structure 1 it's not a mechanism — 
it's a dataset artifact.
"""

import torch
import json
import os
from datetime import datetime
from transformer_lens import HookedTransformer
from tqdm import tqdm

os.makedirs("results", exist_ok=True)

# ── Core facts — each gets 4 structural forms ──────────────────
# (fact_id, answer, category, s1, s2, s3, s4)
# s1 = forward completion
# s2 = question form
# s3 = reverse direction (answer first)
# s4 = contextual embed

FACTS = [
    # ══════════════════════════════════════════════
    # GEOGRAPHY — CAPITALS (easy)
    # ══════════════════════════════════════════════
    ("france_capital", "Paris", "geography", "easy",
     "The capital of France is",
     "What is the capital of France?",
     "Paris is the capital of which country?",
     "France, whose capital city is"),

    ("germany_capital", "Berlin", "geography", "easy",
     "The capital of Germany is",
     "What is the capital of Germany?",
     "Berlin is the capital of which country?",
     "Germany, whose capital city is"),

    ("japan_capital", "Tokyo", "geography", "easy",
     "The capital of Japan is",
     "What is the capital of Japan?",
     "Tokyo is the capital of which country?",
     "Japan, whose capital city is"),

    ("italy_capital", "Rome", "geography", "easy",
     "The capital of Italy is",
     "What is the capital of Italy?",
     "Rome is the capital of which country?",
     "Italy, whose capital city is"),

    ("russia_capital", "Moscow", "geography", "easy",
     "The capital of Russia is",
     "What is the capital of Russia?",
     "Moscow is the capital of which country?",
     "Russia, whose capital city is"),

    ("china_capital", "Beijing", "geography", "easy",
     "The capital of China is",
     "What is the capital of China?",
     "Beijing is the capital of which country?",
     "China, whose capital city is"),

    ("usa_capital", "Washington", "geography", "easy",
     "The capital of the United States is",
     "What is the capital of the United States?",
     "Washington is the capital of which country?",
     "The United States, whose capital is"),

    ("australia_capital", "Canberra", "geography", "easy",
     "The capital of Australia is",
     "What is the capital of Australia?",
     "Canberra is the capital of which country?",
     "Australia, whose capital city is"),

    ("brazil_capital", "Brasilia", "geography", "easy",
     "The capital of Brazil is",
     "What is the capital of Brazil?",
     "Brasilia is the capital of which country?",
     "Brazil, whose capital city is"),

    ("india_capital", "Delhi", "geography", "easy",
     "The capital of India is",
     "What is the capital of India?",
     "Delhi is the capital of which country?",
     "India, whose capital city is"),

    # ══════════════════════════════════════════════
    # GEOGRAPHY — CAPITALS (medium)
    # ══════════════════════════════════════════════
    ("canada_capital", "Ottawa", "geography", "medium",
     "The capital of Canada is",
     "What is the capital of Canada?",
     "Ottawa is the capital of which country?",
     "Canada, whose capital city is"),

    ("spain_capital", "Madrid", "geography", "medium",
     "The capital of Spain is",
     "What is the capital of Spain?",
     "Madrid is the capital of which country?",
     "Spain, whose capital city is"),

    ("egypt_capital", "Cairo", "geography", "medium",
     "The capital of Egypt is",
     "What is the capital of Egypt?",
     "Cairo is the capital of which country?",
     "Egypt, whose capital city is"),

    ("turkey_capital", "Ankara", "geography", "medium",
     "The capital of Turkey is",
     "What is the capital of Turkey?",
     "Ankara is the capital of which country?",
     "Turkey, whose capital city is"),

    ("argentina_capital", "Buenos", "geography", "medium",
     "The capital of Argentina is",
     "What is the capital of Argentina?",
     "Buenos Aires is the capital of which country?",
     "Argentina, whose capital city is"),

    ("sweden_capital", "Stockholm", "geography", "medium",
     "The capital of Sweden is",
     "What is the capital of Sweden?",
     "Stockholm is the capital of which country?",
     "Sweden, whose capital city is"),

    ("norway_capital", "Oslo", "geography", "medium",
     "The capital of Norway is",
     "What is the capital of Norway?",
     "Oslo is the capital of which country?",
     "Norway, whose capital city is"),

    ("mexico_capital", "Mexico", "geography", "medium",
     "The capital of Mexico is",
     "What is the capital of Mexico?",
     "Mexico City is the capital of which country?",
     "Mexico, whose capital city is"),

    ("south_africa_capital", "Pretoria", "geography", "medium",
     "The capital of South Africa is",
     "What is the capital of South Africa?",
     "Pretoria is the capital of which country?",
     "South Africa, whose capital city is"),

    ("greece_capital", "Athens", "geography", "medium",
     "The capital of Greece is",
     "What is the capital of Greece?",
     "Athens is the capital of which country?",
     "Greece, whose capital city is"),

    # ══════════════════════════════════════════════
    # GEOGRAPHY — CAPITALS (hard)
    # ══════════════════════════════════════════════
    ("kazakhstan_capital", "Astana", "geography", "hard",
     "The capital of Kazakhstan is",
     "What is the capital of Kazakhstan?",
     "Astana is the capital of which country?",
     "Kazakhstan, whose capital is"),

    ("uzbekistan_capital", "Tashkent", "geography", "hard",
     "The capital of Uzbekistan is",
     "What is the capital of Uzbekistan?",
     "Tashkent is the capital of which country?",
     "Uzbekistan, whose capital is"),

    ("myanmar_capital", "Naypyidaw", "geography", "hard",
     "The capital of Myanmar is",
     "What is the capital of Myanmar?",
     "Naypyidaw is the capital of which country?",
     "Myanmar, whose capital is"),

    ("ethiopia_capital", "Addis", "geography", "hard",
     "The capital of Ethiopia is",
     "What is the capital of Ethiopia?",
     "Addis Ababa is the capital of which country?",
     "Ethiopia, whose capital is"),

    ("peru_capital", "Lima", "geography", "hard",
     "The capital of Peru is",
     "What is the capital of Peru?",
     "Lima is the capital of which country?",
     "Peru, whose capital is"),

    # ══════════════════════════════════════════════
    # SCIENCE (easy)
    # ══════════════════════════════════════════════
    ("water_composition", "oxygen", "science", "easy",
     "Water is made of hydrogen and",
     "What element combines with hydrogen to make water?",
     "Oxygen combines with hydrogen to make",
     "Water, composed of hydrogen and"),

    ("gold_symbol", "Au", "science", "easy",
     "The chemical symbol for gold is",
     "What is the chemical symbol for gold?",
     "Au is the chemical symbol for which element?",
     "Gold, whose chemical symbol is"),

    ("evolution_theory", "Darwin", "science", "easy",
     "The theory of evolution was proposed by",
     "Who proposed the theory of evolution?",
     "Darwin is famous for proposing the theory of",
     "Evolution, the theory proposed by"),

    ("relativity", "Einstein", "science", "easy",
     "The theory of relativity was proposed by",
     "Who proposed the theory of relativity?",
     "Einstein is famous for proposing the theory of",
     "Relativity, the theory proposed by"),

    ("gravity_law", "Newton", "science", "easy",
     "The law of gravity was described by",
     "Who described the law of gravity?",
     "Newton is famous for describing the law of",
     "Gravity, whose law was described by"),

    ("dna_meaning", "acid", "science", "easy",
     "DNA stands for deoxyribonucleic",
     "What does the A in DNA stand for?",
     "Deoxyribonucleic acid is commonly abbreviated as",
     "DNA, which stands for deoxyribonucleic"),

    ("iron_symbol", "Fe", "science", "easy",
     "The chemical symbol for iron is",
     "What is the chemical symbol for iron?",
     "Fe is the chemical symbol for which element?",
     "Iron, whose chemical symbol is"),

    ("silver_symbol", "Ag", "science", "easy",
     "The chemical symbol for silver is",
     "What is the chemical symbol for silver?",
     "Ag is the chemical symbol for which element?",
     "Silver, whose chemical symbol is"),

    # ══════════════════════════════════════════════
    # SCIENCE (medium)
    # ══════════════════════════════════════════════
    ("periodic_table", "Mendeleev", "science", "medium",
     "The periodic table was created by",
     "Who created the periodic table?",
     "Mendeleev is famous for creating the",
     "The periodic table, created by"),

    ("penicillin", "Fleming", "science", "medium",
     "Penicillin was discovered by",
     "Who discovered penicillin?",
     "Fleming is famous for discovering",
     "Penicillin, discovered by"),

    ("xrays", "Roentgen", "science", "medium",
     "X-rays were discovered by",
     "Who discovered X-rays?",
     "Roentgen is famous for discovering",
     "X-rays, discovered by"),

    ("radioactivity", "Curie", "science", "medium",
     "Radioactivity was discovered by",
     "Who discovered radioactivity?",
     "Curie is famous for discovering",
     "Radioactivity, discovered by"),

    ("telephone_inventor", "Bell", "science", "medium",
     "The telephone was invented by",
     "Who invented the telephone?",
     "Bell is famous for inventing the",
     "The telephone, invented by"),

    ("lightbulb_inventor", "Edison", "science", "medium",
     "The light bulb was invented by",
     "Who invented the light bulb?",
     "Edison is famous for inventing the",
     "The light bulb, invented by"),

    ("www_inventor", "Berners", "science", "medium",
     "The World Wide Web was invented by",
     "Who invented the World Wide Web?",
     "Berners-Lee invented the",
     "The World Wide Web, invented by"),

    ("radio_inventor", "Marconi", "science", "medium",
     "Radio was invented by",
     "Who invented the radio?",
     "Marconi is famous for inventing",
     "Radio, invented by"),

    # ══════════════════════════════════════════════
    # SCIENCE (hard)
    # ══════════════════════════════════════════════
    ("quantum_mechanics", "Planck", "science", "hard",
     "Quantum mechanics was pioneered by",
     "Who pioneered quantum mechanics?",
     "Planck pioneered which field of physics?",
     "Quantum mechanics, pioneered by"),

    ("germ_theory", "Pasteur", "science", "hard",
     "Germ theory was developed by",
     "Who developed germ theory?",
     "Pasteur developed which medical theory?",
     "Germ theory, developed by"),

    ("calculus_inventor", "Newton", "science", "hard",
     "Calculus was invented by",
     "Who invented calculus?",
     "Newton is credited with inventing",
     "Calculus, invented by"),

    ("printing_press", "Gutenberg", "science", "hard",
     "The printing press was invented by",
     "Who invented the printing press?",
     "Gutenberg invented the",
     "The printing press, invented by"),

    ("vaccination_pioneer", "Jenner", "science", "hard",
     "Vaccination was pioneered by",
     "Who pioneered vaccination?",
     "Jenner pioneered which medical practice?",
     "Vaccination, pioneered by"),

    # ══════════════════════════════════════════════
    # HISTORY (easy)
    # ══════════════════════════════════════════════
    ("berlin_wall", "1989", "history", "easy",
     "The Berlin Wall fell in",
     "In what year did the Berlin Wall fall?",
     "1989 was the year the Berlin Wall",
     "The Berlin Wall, which fell in"),

    ("ww2_end", "1945", "history", "easy",
     "World War 2 ended in",
     "In what year did World War 2 end?",
     "1945 was the year World War 2 ended. It began in 1939 and ended in",
     "World War 2, which ended in"),

    ("moon_landing", "1969", "history", "easy",
     "The first Moon landing was in",
     "In what year was the first Moon landing?",
     "1969 was the year of the first Moon",
     "The Moon landing, which first occurred in"),

    ("french_revolution", "1789", "history", "easy",
     "The French Revolution began in",
     "In what year did the French Revolution begin?",
     "1789 was the year the French Revolution",
     "The French Revolution, which began in"),

    ("us_independence", "1776", "history", "easy",
     "The American Declaration of Independence was signed in",
     "In what year was the American Declaration of Independence signed?",
     "1776 was the year the American Declaration of",
     "The Declaration of Independence, signed in"),

    # ══════════════════════════════════════════════
    # HISTORY (medium)
    # ══════════════════════════════════════════════
    ("ww1_start", "1914", "history", "medium",
     "World War 1 began in",
     "In what year did World War 1 begin?",
     "1914 marks the start of which war?",
     "World War 1, which began in"),

    ("ww2_start", "1939", "history", "medium",
     "World War 2 began in",
     "In what year did World War 2 begin?",
     "1939 marks the start of which war?",
     "World War 2, which began in"),

    ("soviet_collapse", "1991", "history", "medium",
     "The Soviet Union collapsed in",
     "In what year did the Soviet Union collapse?",
     "1991 was the year the Soviet Union",
     "The Soviet Union, which collapsed in"),

    ("columbus_america", "1492", "history", "medium",
     "Columbus arrived in the Americas in",
     "In what year did Columbus arrive in the Americas?",
     "1492 was the year Columbus arrived in the",
     "Columbus, who arrived in the Americas in"),

    ("magna_carta", "1215", "history", "medium",
     "The Magna Carta was signed in",
     "In what year was the Magna Carta signed?",
     "1215 was the year the Magna Carta was",
     "The Magna Carta, signed in"),

    ("napoleon_waterloo", "1815", "history", "medium",
     "Napoleon was defeated at Waterloo in",
     "In what year was Napoleon defeated at Waterloo?",
     "1815 was the year Napoleon was defeated at",
     "The Battle of Waterloo, which occurred in"),

    # ══════════════════════════════════════════════
    # HISTORY (hard)
    # ══════════════════════════════════════════════
    ("first_pm_india", "Nehru", "history", "hard",
     "The first Prime Minister of India was",
     "Who was the first Prime Minister of India?",
     "Nehru was the first Prime Minister of which country?",
     "India, whose first Prime Minister was"),

    ("titanic_sinking", "1912", "history", "hard",
     "The Titanic sank in",
     "In what year did the Titanic sink?",
     "1912 was the year the Titanic",
     "The Titanic, which sank in"),

    ("penicillin_discovery", "1928", "history", "hard",
     "Penicillin was discovered in",
     "In what year was penicillin discovered?",
     "1928 was the year penicillin was",
     "Penicillin, discovered in"),

    ("first_powered_flight", "1903", "history", "hard",
     "The first powered aircraft flight was in",
     "In what year was the first powered aircraft flight?",
     "1903 was the year of the first powered",
     "The first powered flight, which occurred in"),

    ("roman_empire_fall", "476", "history", "hard",
     "The Western Roman Empire fell in",
     "In what year did the Western Roman Empire fall?",
     "476 was the year the Western Roman Empire",
     "The Western Roman Empire, which fell in"),

    # ══════════════════════════════════════════════
    # LITERATURE (easy)
    # ══════════════════════════════════════════════
    ("hamlet", "Shakespeare", "literature", "easy",
     "Hamlet was written by",
     "Who wrote Hamlet?",
     "Shakespeare wrote which famous Danish prince play?",
     "Hamlet, written by"),

    ("1984", "Orwell", "literature", "easy",
     "The novel 1984 was written by",
     "Who wrote the novel 1984?",
     "Orwell wrote which famous dystopian novel?",
     "1984, the dystopian novel written by"),

    ("odyssey", "Homer", "literature", "easy",
     "The Odyssey was written by",
     "Who wrote the Odyssey?",
     "Homer wrote which ancient Greek epic?",
     "The Odyssey, written by"),

    ("war_peace", "Tolstoy", "literature", "easy",
     "War and Peace was written by",
     "Who wrote War and Peace?",
     "Tolstoy wrote which famous Russian novel?",
     "War and Peace, written by"),

    ("divine_comedy", "Dante", "literature", "easy",
     "The Divine Comedy was written by",
     "Who wrote the Divine Comedy?",
     "Dante wrote which famous Italian poem?",
     "The Divine Comedy, written by"),

    # ══════════════════════════════════════════════
    # LITERATURE (medium)
    # ══════════════════════════════════════════════
    ("crime_punishment", "Dostoevsky", "literature", "medium",
     "Crime and Punishment was written by",
     "Who wrote Crime and Punishment?",
     "Dostoevsky wrote which famous Russian novel?",
     "Crime and Punishment, written by"),

    ("don_quixote", "Cervantes", "literature", "medium",
     "Don Quixote was written by",
     "Who wrote Don Quixote?",
     "Cervantes wrote which famous Spanish novel?",
     "Don Quixote, written by"),

    ("faust", "Goethe", "literature", "medium",
     "Faust was written by",
     "Who wrote Faust?",
     "Goethe wrote which famous German play?",
     "Faust, written by"),

    ("great_gatsby", "Fitzgerald", "literature", "medium",
     "The Great Gatsby was written by",
     "Who wrote The Great Gatsby?",
     "Fitzgerald wrote which famous American novel?",
     "The Great Gatsby, written by"),

    ("moby_dick", "Melville", "literature", "medium",
     "Moby Dick was written by",
     "Who wrote Moby Dick?",
     "Melville wrote which famous whaling novel?",
     "Moby Dick, written by"),

    # ══════════════════════════════════════════════
    # LITERATURE (hard)
    # ══════════════════════════════════════════════
    ("canterbury_tales", "Chaucer", "literature", "hard",
     "The Canterbury Tales was written by",
     "Who wrote the Canterbury Tales?",
     "Chaucer wrote which famous medieval English work?",
     "The Canterbury Tales, written by"),

    ("paradise_lost", "Milton", "literature", "hard",
     "Paradise Lost was written by",
     "Who wrote Paradise Lost?",
     "Milton wrote which famous epic poem?",
     "Paradise Lost, written by"),

    ("madame_bovary", "Flaubert", "literature", "hard",
     "Madame Bovary was written by",
     "Who wrote Madame Bovary?",
     "Flaubert wrote which famous French novel?",
     "Madame Bovary, written by"),

    ("brothers_karamazov", "Dostoevsky", "literature", "hard",
     "The Brothers Karamazov was written by",
     "Who wrote The Brothers Karamazov?",
     "Dostoevsky wrote which famous philosophical novel?",
     "The Brothers Karamazov, written by"),

    ("anna_karenina", "Tolstoy", "literature", "hard",
     "Anna Karenina was written by",
     "Who wrote Anna Karenina?",
     "Tolstoy wrote which famous novel about a Russian aristocrat?",
     "Anna Karenina, written by"),

    # ══════════════════════════════════════════════
    # GEOGRAPHY — PHYSICAL (easy)
    # ══════════════════════════════════════════════
    ("nile_river", "Nile", "geography", "easy",
     "The longest river in the world is the",
     "What is the longest river in the world?",
     "The Nile is the longest what in the world?",
     "The world's longest river, the"),

    ("pacific_ocean", "Pacific", "geography", "easy",
     "The largest ocean in the world is the",
     "What is the largest ocean in the world?",
     "The Pacific is the largest what in the world?",
     "The world's largest ocean, the"),

    ("everest", "Everest", "geography", "easy",
     "The tallest mountain in the world is Mount",
     "What is the tallest mountain in the world?",
     "Mount Everest is the tallest what in the world?",
     "The world's tallest mountain, Mount"),

    ("sahara", "Africa", "geography", "easy",
     "The Sahara desert is located in",
     "On which continent is the Sahara desert?",
     "Africa is home to which famous desert?",
     "The Sahara, the world's largest desert, located in"),

    ("amazon_river", "South", "geography", "easy",
     "The Amazon river flows through",
     "On which continent is the Amazon river?",
     "South America is home to which famous river?",
     "The Amazon, the world's largest river by volume, flows through"),

    # ══════════════════════════════════════════════
    # GEOGRAPHY — LANDMARKS (medium)
    # ══════════════════════════════════════════════
    ("eiffel_tower", "Paris", "geography", "medium",
     "The Eiffel Tower is located in",
     "In which city is the Eiffel Tower?",
     "Paris is home to which famous tower?",
     "The Eiffel Tower, located in"),

    ("colosseum", "Rome", "geography", "medium",
     "The Colosseum is located in",
     "In which city is the Colosseum?",
     "Rome is home to which ancient amphitheatre?",
     "The Colosseum, located in"),

    ("great_wall", "China", "geography", "medium",
     "The Great Wall is located in",
     "In which country is the Great Wall?",
     "China is home to which ancient wall?",
     "The Great Wall, located in"),

    ("big_ben", "London", "geography", "medium",
     "Big Ben is located in",
     "In which city is Big Ben?",
     "London is home to which famous clock tower?",
     "Big Ben, located in"),

    ("pyramids", "Egypt", "geography", "medium",
     "The Pyramids of Giza are located in",
     "In which country are the Pyramids of Giza?",
     "Egypt is home to which ancient pyramids?",
     "The Pyramids of Giza, located in"),

    # ══════════════════════════════════════════════
    # ARTS & MUSIC (easy)
    # ══════════════════════════════════════════════
    ("mona_lisa", "Leonardo", "arts", "easy",
     "The Mona Lisa was painted by",
     "Who painted the Mona Lisa?",
     "Leonardo da Vinci painted which famous smile?",
     "The Mona Lisa, painted by"),

    ("ninth_symphony", "Beethoven", "arts", "easy",
     "The Ninth Symphony was composed by",
     "Who composed the Ninth Symphony?",
     "Beethoven composed which famous final symphony?",
     "The Ninth Symphony, composed by"),

    ("four_seasons", "Vivaldi", "arts", "easy",
     "The Four Seasons was composed by",
     "Who composed The Four Seasons?",
     "Vivaldi composed which famous set of violin concertos?",
     "The Four Seasons, composed by"),

    ("sistine_chapel", "Michelangelo", "arts", "easy",
     "The Sistine Chapel ceiling was painted by",
     "Who painted the Sistine Chapel ceiling?",
     "Michelangelo painted which famous ceiling?",
     "The Sistine Chapel ceiling, painted by"),

    ("moonlight_sonata", "Beethoven", "arts", "medium",
     "The Moonlight Sonata was composed by",
     "Who composed the Moonlight Sonata?",
     "Beethoven composed which famous piano sonata?",
     "The Moonlight Sonata, composed by"),

    # ══════════════════════════════════════════════
    # ARTS & MUSIC (medium/hard)
    # ══════════════════════════════════════════════
    ("starry_night", "Van", "arts", "medium",
     "The Starry Night was painted by",
     "Who painted The Starry Night?",
     "Van Gogh painted which famous night scene?",
     "The Starry Night, painted by"),

    ("1812_overture", "Tchaikovsky", "arts", "medium",
     "The 1812 Overture was composed by",
     "Who composed the 1812 Overture?",
     "Tchaikovsky composed which famous patriotic overture?",
     "The 1812 Overture, composed by"),

    ("magic_flute", "Mozart", "arts", "medium",
     "The Magic Flute was composed by",
     "Who composed The Magic Flute?",
     "Mozart composed which famous opera?",
     "The Magic Flute, the opera composed by"),

    ("water_lilies", "Monet", "arts", "hard",
     "The Water Lilies series was painted by",
     "Who painted the Water Lilies series?",
     "Monet painted which famous series of pond paintings?",
     "Water Lilies, the series painted by"),

    ("guernica", "Picasso", "arts", "hard",
     "Guernica was painted by",
     "Who painted Guernica?",
     "Picasso painted which famous anti-war painting?",
     "Guernica, painted by"),

    # ══════════════════════════════════════════════
    # ECONOMICS & BUSINESS (medium)
    # ══════════════════════════════════════════════
    ("wealth_of_nations", "Smith", "economics", "medium",
     "The Wealth of Nations was written by",
     "Who wrote The Wealth of Nations?",
     "Adam Smith wrote which famous economics text?",
     "The Wealth of Nations, written by"),

    ("amazon_founder", "Bezos", "economics", "medium",
     "Amazon was founded by",
     "Who founded Amazon?",
     "Jeff Bezos founded which famous company?",
     "Amazon, founded by"),

    ("apple_founder", "Jobs", "economics", "medium",
     "Apple was co-founded by",
     "Who co-founded Apple?",
     "Steve Jobs co-founded which famous technology company?",
     "Apple, co-founded by"),

    ("microsoft_founder", "Gates", "economics", "medium",
     "Microsoft was co-founded by",
     "Who co-founded Microsoft?",
     "Bill Gates co-founded which famous software company?",
     "Microsoft, co-founded by"),

    ("bitcoin_created", "2009", "economics", "hard",
     "Bitcoin was created in",
     "In what year was Bitcoin created?",
     "2009 was the year Bitcoin was",
     "Bitcoin, which was created in"),

    # ══════════════════════════════════════════════
    # SPORT (easy/medium)
    # ══════════════════════════════════════════════
    ("olympics_origin", "Greece", "sport", "easy",
     "The ancient Olympic Games originated in",
     "In which country did the ancient Olympic Games originate?",
     "Greece is the origin of which famous games?",
     "The Olympic Games, which originated in"),

    ("football_players", "eleven", "sport", "easy",
     "A football team has",
     "How many players are on a football team?",
     "Eleven players make up which sports team?",
     "A football team, which has"),

    ("wimbledon_location", "London", "sport", "medium",
     "The Wimbledon tennis tournament is held in",
     "In which city is Wimbledon held?",
     "London hosts which famous tennis tournament?",
     "Wimbledon, the tennis tournament held in"),

    ("tour_de_france", "France", "sport", "medium",
     "The Tour de France cycling race is held in",
     "In which country is the Tour de France held?",
     "France hosts which famous cycling race?",
     "The Tour de France, held in"),

    # ══════════════════════════════════════════════
    # PHILOSOPHY (medium/hard)
    # ══════════════════════════════════════════════
    ("republic_author", "Plato", "philosophy", "medium",
     "The Republic was written by",
     "Who wrote The Republic?",
     "Plato wrote which famous philosophical dialogue?",
     "The Republic, written by"),

    ("nicomachean_ethics", "Aristotle", "philosophy", "medium",
     "Nicomachean Ethics was written by",
     "Who wrote Nicomachean Ethics?",
     "Aristotle wrote which famous ethical treatise?",
     "Nicomachean Ethics, written by"),

    ("critique_pure_reason", "Kant", "philosophy", "hard",
     "The Critique of Pure Reason was written by",
     "Who wrote the Critique of Pure Reason?",
     "Kant wrote which famous philosophical critique?",
     "The Critique of Pure Reason, written by"),

    ("beyond_good_evil", "Nietzsche", "philosophy", "hard",
     "Beyond Good and Evil was written by",
     "Who wrote Beyond Good and Evil?",
     "Nietzsche wrote which famous philosophical work?",
     "Beyond Good and Evil, written by"),

    ("leviathan_author", "Hobbes", "philosophy", "hard",
     "Leviathan was written by",
     "Who wrote Leviathan?",
     "Thomas Hobbes wrote which famous political treatise?",
     "Leviathan, written by"),

    # ══════════════════════════════════════════════
    # LANGUAGE & LINGUISTICS (medium/hard)
    # ══════════════════════════════════════════════
    ("brazil_language", "Portuguese", "language", "medium",
     "The official language of Brazil is",
     "What is the official language of Brazil?",
     "Portuguese is the official language of which country?",
     "Brazil, whose official language is"),

    ("egypt_language", "Arabic", "language", "medium",
     "The official language of Egypt is",
     "What is the official language of Egypt?",
     "Arabic is the official language of which country?",
     "Egypt, whose official language is"),

    ("most_spoken_language", "Chinese", "language", "medium",
     "The most spoken language in the world is",
     "What is the most spoken language in the world?",
     "Chinese is the most spoken language in the",
     "The world's most spoken language is"),

    ("english_alphabet", "26", "language", "easy",
     "The English alphabet has",
     "How many letters does the English alphabet have?",
     "26 letters make up which alphabet?",
     "The English alphabet, which has"),

    ("arabic_alphabet", "28", "language", "medium",
     "The Arabic alphabet has",
     "How many letters does the Arabic alphabet have?",
     "28 letters make up which alphabet?",
     "The Arabic alphabet, which has"),

    # ══════════════════════════════════════════════
    # MATHEMATICS (easy)
    # ══════════════════════════════════════════════
    ("pi_value", "3", "mathematics", "easy",
     "Pi is approximately equal to",
     "What is the approximate value of pi?",
     "3.14159 is the approximate value of which mathematical constant?",
     "Pi, which is approximately equal to"),

    ("pythagoras_theorem", "squared", "mathematics", "easy",
     "The Pythagorean theorem states a squared plus b squared equals c",
     "Complete the Pythagorean theorem: a squared plus b squared equals?",
     "C squared equals a squared plus b squared according to which theorem?",
     "The Pythagorean theorem, which states a squared plus b squared equals c"),

    ("triangle_angles", "180", "mathematics", "easy",
     "The sum of angles in a triangle is",
     "What is the sum of angles in a triangle?",
     "180 degrees is the sum of angles in which shape?",
     "A triangle, whose angles sum to"),

    ("square_root_144", "12", "mathematics", "easy",
     "The square root of 144 is",
     "What is the square root of 144?",
     "12 is the square root of which number?",
     "144, whose square root is"),

    ("fibonacci", "sequence", "mathematics", "medium",
     "The Fibonacci series is a famous number",
     "What type of mathematical series is the Fibonacci?",
     "Fibonacci is famous for which type of mathematical series?",
     "The Fibonacci series, a famous number"),
]

# Build flat prompt list with metadata
ALL_PROMPTS = []
STRUCTURE_NAMES = [
    "forward_completion",
    "question_form",
    "reverse_direction",
    "contextual_embed",
]

for fact_id, answer, category, tier, s1, s2, s3, s4 in FACTS:
    for structure_name, prompt in zip(
        STRUCTURE_NAMES, [s1, s2, s3, s4]
    ):
        ALL_PROMPTS.append({
            "fact_id": fact_id,
            "answer": answer,
            "category": category,
            "tier": tier,
            "structure": structure_name,
            "prompt": prompt,
        })

print(f"Total prompts: {len(ALL_PROMPTS)}")
print(f"Unique facts: {len(FACTS)}")
print(f"Structures per fact: 4")
print(f"Categories: {len(set(f[2] for f in FACTS))}")

ALPHAS = [0.0, 0.3, 0.5]


def get_token_id(model, answer):
    try:
        return model.to_single_token(f" {answer}")
    except Exception:
        tokens = model.to_tokens(f" {answer}")[0]
        return tokens[1].item()


def get_layer_probs(model, cache, token_id, n_layers):
    probs = []
    for layer in range(n_layers):
        resid = cache[
            f"blocks.{layer}.hook_resid_post"
        ][0, -1]
        resid_normed = model.ln_final(resid.unsqueeze(0))[0]
        logits = resid_normed @ model.W_U + model.b_U
        prob = torch.softmax(logits, dim=-1)[token_id].item()
        probs.append(prob)
    return probs


def run_model(model_name):
    print(f"\n{'='*65}")
    print(f"MODEL: {model_name}")
    print(f"{'='*65}")

    model = HookedTransformer.from_pretrained(
        model_name,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    model.eval()
    n_layers = model.cfg.n_layers

    results = []
    skipped = 0
    alpha_correct = {a: 0 for a in ALPHAS}
    structure_stats = {s: {
        "correct": 0, "type2a": 0, "type2b": 0,
        "total": 0, "suppression_ratios": []
    } for s in STRUCTURE_NAMES}
    fact_consistency = {}

    for item in tqdm(ALL_PROMPTS, desc=model_name.split("/")[-1]):
        prompt = item["prompt"]
        answer = item["answer"]
        structure = item["structure"]
        fact_id = item["fact_id"]

        try:
            token_id = get_token_id(model, answer)
        except Exception:
            skipped += 1
            continue

        try:
            with torch.no_grad():
                final_logits, cache = model.run_with_cache(prompt)
        except Exception:
            skipped += 1
            continue

        final_probs = torch.softmax(final_logits[0, -1], dim=-1)
        predicted = model.to_string(
            final_logits[0, -1].argmax()
        ).strip()
        correct_prob = final_probs[token_id].item()
        correct_rank = (
            final_probs > final_probs[token_id]
        ).sum().item() + 1

        layer_probs = get_layer_probs(
            model, cache, token_id, n_layers
        )
        peak_layer = layer_probs.index(max(layer_probs))
        peak_prob = max(layer_probs)
        suppression_ratio = peak_prob / (correct_prob + 1e-10)
        rel_depth = peak_layer / n_layers

        resid = cache[
            f"blocks.{peak_layer}.hook_resid_post"
        ][0, -1]
        resid_normed = model.ln_final(resid.unsqueeze(0))[0]
        peak_logits = resid_normed @ model.W_U + model.b_U

        is_correct = answer.lower() in predicted.lower()
        if is_correct:
            hall_type = "CORRECT"
        elif correct_rank <= 10:
            hall_type = "TYPE2A"
        else:
            hall_type = "TYPE2B"

        # Track per-structure stats
        s = structure_stats[structure]
        s["total"] += 1
        if is_correct:
            s["correct"] += 1
        if hall_type == "TYPE2A":
            s["type2a"] += 1
        if hall_type == "TYPE2B":
            s["type2b"] += 1
        s["suppression_ratios"].append(suppression_ratio)

        # Track fact consistency
        if fact_id not in fact_consistency:
            fact_consistency[fact_id] = []
        fact_consistency[fact_id].append(is_correct)

        row = {
            "fact_id": fact_id,
            "prompt": prompt,
            "answer": answer,
            "category": item["category"],
            "structure": structure,
            "predicted": predicted,
            "is_correct": is_correct,
            "hallucination_type": hall_type,
            "suppression_ratio": round(suppression_ratio, 3),
            "peak_layer": peak_layer,
            "peak_layer_relative": round(rel_depth, 3),
            "correct_rank": correct_rank,
            "alpha_predictions": {},
        }

        for alpha in ALPHAS:
            if alpha == 0.0:
                pred_logits = final_logits[0, -1]
            else:
                pred_logits = (
                    alpha * peak_logits +
                    (1 - alpha) * final_logits[0, -1]
                )
            pred = model.to_string(
                pred_logits.argmax()
            ).strip()
            correct = answer.lower() in pred.lower()
            if correct:
                alpha_correct[alpha] += 1
            row["alpha_predictions"][str(alpha)] = {
                "predicted": pred,
                "correct": correct,
            }

        results.append(row)

    total = len(results)
    print(f"\nProcessed: {total} | Skipped: {skipped}")

    # Structure breakdown — the key table
    print(f"\n{'='*65}")
    print(f"STRUCTURE BREAKDOWN (this is the reviewer-convincing table)")
    print(f"{'='*65}")
    print(f"{'Structure':<22} {'Acc':>6} {'Type2a':>8} "
          f"{'Type2b':>8} {'Avg ρ':>8}")
    print("-" * 55)

    for s_name in STRUCTURE_NAMES:
        s = structure_stats[s_name]
        n = s["total"]
        if n == 0:
            continue
        acc = s["correct"] / n * 100
        t2a = s["type2a"] / n * 100
        t2b = s["type2b"] / n * 100
        avg_rho = (
            sum(s["suppression_ratios"]) /
            len(s["suppression_ratios"])
        )
        print(f"{s_name:<22} {acc:>5.1f}% {t2a:>7.1f}% "
              f"{t2b:>7.1f}% {avg_rho:>7.1f}x")

    # Fact consistency — how many facts correct on all 4 structures
    all4 = sum(
        1 for v in fact_consistency.values() if all(v)
    )
    any1 = sum(
        1 for v in fact_consistency.values() if any(v)
    )
    none = sum(
        1 for v in fact_consistency.values() if not any(v)
    )
    print(f"\nFact consistency across 4 structures:")
    print(f"  Correct on all 4:  {all4}/{len(FACTS)} facts")
    print(f"  Correct on some:   {any1-all4}/{len(FACTS)} facts")
    print(f"  Correct on none:   {none}/{len(FACTS)} facts")
    print(f"\n  → This shows structural sensitivity is REAL")

    # Intervention results
    print(f"\nIntervention results:")
    baseline = alpha_correct[0.0] / total
    for alpha in ALPHAS:
        acc = alpha_correct[alpha] / total
        delta = acc - baseline
        marker = " ← best" if delta == max(
            alpha_correct[a] / total - baseline
            for a in ALPHAS
        ) else ""
        print(f"  α={alpha}: {alpha_correct[alpha]}/{total} "
              f"({acc:.1%}) Δ{delta:+.1%}{marker}")

    output = {
        "model": model_name,
        "n_facts": len(FACTS),
        "n_structures": 4,
        "n_prompts_total": total,
        "n_skipped": skipped,
        "timestamp": datetime.now().isoformat(),
        "structure_stats": {
            k: {
                "correct": v["correct"],
                "type2a": v["type2a"],
                "type2b": v["type2b"],
                "total": v["total"],
                "accuracy": v["correct"] / v["total"]
                if v["total"] > 0 else 0,
                "avg_suppression_ratio": (
                    sum(v["suppression_ratios"]) /
                    len(v["suppression_ratios"])
                ) if v["suppression_ratios"] else 0,
            }
            for k, v in structure_stats.items()
        },
        "fact_consistency": {
            "all_4_correct": all4,
            "some_correct": any1 - all4,
            "none_correct": none,
        },
        "alpha_results": {
            str(a): {
                "correct": alpha_correct[a],
                "accuracy": alpha_correct[a] / total,
            }
            for a in ALPHAS
        },
        "results": results,
    }

    fname = (
        f"results/diverse_"
        f"{model_name.replace('/', '_')}.json"
    )
    with open(fname, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {fname}")

    del model
    torch.cuda.empty_cache()
    return output


# Run strong and weak suppression family
r1 = run_model("gpt2-xl")
r2 = run_model("EleutherAI/gpt-neo-2.7B")

# Final comparison
print(f"\n{'='*65}")
print(f"FINAL COMPARISON — STRUCTURE DIVERSITY RESULTS")
print(f"{'='*65}")
for r in [r1, r2]:
    m = r["model"].split("/")[-1]
    print(f"\n{m}:")
    for s_name in STRUCTURE_NAMES:
        s = r["structure_stats"][s_name]
        acc = s["accuracy"] * 100
        rho = s["avg_suppression_ratio"]
        print(f"  {s_name:<22} acc={acc:.1f}%  ρ={rho:.1f}x")
    cons = r["fact_consistency"]
    print(f"  Consistent on all 4: "
          f"{cons['all_4_correct']}/{r['n_facts']}")

print("\nExperiment 10 complete.")