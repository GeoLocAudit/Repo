"""Write data/questions.csv: 300 questions = 10 topics x 3 groups x 10.

Groups (no question names a place; the location comes only from the API setting):
  explicit   asks for something nearby ("near me", "in my area")
  implicit   the right answer depends on where the user is (prices, laws, providers, weather)
  nonlocal   the answer should not depend on the user's location
The first question of every topic and group is in the pilot (30 questions).

  python scripts/make_questions.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geoaudit.common import DATA  # noqa: E402

Q = {
 'home services': {
  'explicit': ['best plumber near me', 'emergency electrician near me open now', 'reliable roofing companies in my area',
               'HVAC repair near me with good reviews', 'house cleaning services near me', 'pest control companies nearby',
               'landscapers near me for small yards', 'local movers with good reviews', 'handyman services near me',
               'appliance repair near me'],
  'implicit': ['how much does it cost to replace a roof', 'do I need a permit to build a deck',
               'how much does a plumber charge per hour', 'when is the best time to plant tomatoes',
               'what is the average cost of installing central air', 'do I need a licensed electrician to replace a light fixture',
               'how much does house cleaning cost', 'is it worth installing solar panels on my house',
               'how much does it cost to move a two bedroom apartment', 'what kind of grass grows best for a lawn'],
  'nonlocal': ['how to unclog a drain without chemicals', 'best cordless drill', 'how to paint a room',
               'best robot vacuum', 'how to fix a running toilet', 'best mattress for back pain',
               'how to remove mold from a shower', 'best air purifier for allergies', 'how to choose a water heater',
               'best smart thermostat']},
 'health care': {
  'explicit': ['urgent care near me open now', 'where can I get a flu shot near me', 'dentists near me accepting new patients',
               'pediatricians in my area', 'physical therapists near me', 'dermatologists near me with short wait times',
               'where can I get an eye exam near me', 'walk-in clinic near me', 'mental health counselors in my area',
               '24 hour pharmacy near me'],
  'implicit': ['how much does an MRI cost without insurance', 'which health insurance plans cover the most doctors',
               'is the flu bad this season', 'how do I get a prescription for birth control',
               'what vaccines are required for school', 'can a pharmacist prescribe antibiotics',
               'how much does a dentist visit cost without insurance', 'is tap water safe to drink',
               'what is the air quality today', 'how long is the wait to see a specialist'],
  'nonlocal': ['how much water should I drink per day', 'best running shoes for flat feet',
               'what are the symptoms of vitamin D deficiency', 'best fitness tracker', 'how to improve sleep quality',
               'best electric toothbrush', 'how long does a cold last', 'best protein powder',
               'how to lower cholesterol naturally', 'best meditation app']},
 'personal finance': {
  'explicit': ['banks near me with no monthly fees', 'credit unions in my area', 'financial advisors near me',
               'tax preparers near me', 'currency exchange near me', 'best mortgage lenders in my area',
               'ATMs near me without fees', 'where to open a business bank account near me',
               'coin counting machine near me', 'notary public near me'],
  'implicit': ['how much is sales tax on a car', 'what is the income tax rate', 'best bank for a savings account',
               'what credit score do I need to buy a house', 'how much is property tax on a home',
               'best credit card for everyday spending', 'how can I file my taxes for free', 'what is the minimum wage',
               'what is the average mortgage rate right now', 'how much can I contribute to a retirement account this year'],
  'nonlocal': ['what is an index fund', 'best budgeting app', 'how does compound interest work',
               'best investment apps for beginners', 'how to build an emergency fund', 'best personal finance books',
               'what is a credit utilization ratio', 'how to pay off debt faster', 'best cryptocurrency wallet',
               'how does a high yield savings account work']},
 'insurance': {
  'explicit': ['local insurance agents near me', 'cheapest car insurance in my area',
               'home insurance companies that serve my area', 'health insurance brokers near me',
               'renters insurance companies in my area', 'independent insurance agency near me',
               'life insurance agents near me', 'flood insurance providers in my area',
               'motorcycle insurance agents near me', 'business insurance brokers near me'],
  'implicit': ['how much is car insurance per month', 'is flood insurance required for my home',
               'what is the cheapest health insurance option', 'how much does home insurance cost',
               'what is the minimum car insurance required', 'best car insurance company',
               'how much is health insurance for a family of four', 'is earthquake insurance worth it',
               'how much does renters insurance cost', 'do I need health insurance by law'],
  'nonlocal': ['what is the difference between term and whole life insurance', 'what does comprehensive car insurance cover',
               'what is an insurance deductible', 'how are insurance premiums calculated', 'is travel insurance worth it',
               'best pet insurance', 'what is umbrella insurance', 'does car insurance cover rental cars',
               'what is gap insurance', 'best life insurance companies']},
 'telecom and utilities': {
  'explicit': ['best internet providers in my area', 'fiber internet available near me', 'phone repair shop near me',
               'which cell carrier has the best coverage in my area', 'cable TV providers near me',
               'electricity providers in my area', 'solar installers near me', 'where to recycle electronics near me',
               'mobile phone store near me', 'water delivery service near me'],
  'implicit': ['which internet provider is best for working from home', 'how much does home internet cost',
               'which mobile carrier has the best coverage', 'what is the cheapest phone plan',
               'how much is the average electricity bill', 'best streaming service for live sports',
               'can I switch electricity providers', 'how fast is the average home internet connection',
               'best prepaid phone plan', 'is 5G home internet available'],
  'nonlocal': ['best wifi router for a large house', 'how to speed up wifi', 'best noise cancelling headphones',
               'best laptop for students', 'iPhone or Android which is better', 'best mesh wifi system',
               'how much internet speed do I need for streaming', 'best VPN', 'best smartphone camera',
               'best tablet for reading']},
 'food and dining': {
  'explicit': ['best pizza near me', 'good Italian restaurants nearby', 'brunch spots near me', 'vegan restaurants near me',
               'coffee shops near me with wifi', 'best sushi in my area', 'restaurants open late near me',
               'bakeries near me', 'farmers markets near me this weekend', 'food delivery services available in my area'],
  'implicit': ['what is the local food specialty I should try', 'how much should I tip at a restaurant',
               'what is the best food delivery app', 'can I buy alcohol on Sunday',
               'which grocery store has the lowest prices', 'when are farmers markets in season',
               'best meal kit delivery service', 'what time do restaurants usually close',
               'can I order alcohol for delivery', 'which supermarket has the best quality produce'],
  'nonlocal': ['easy weeknight dinner recipes', 'best chef knife', 'how to make sourdough bread', 'best coffee maker',
               'healthy breakfast ideas', 'best air fryer', 'how to cook rice perfectly', 'best non stick pan',
               'vegetarian sources of protein', 'best blender for smoothies']},
 'retail': {
  'explicit': ['grocery stores near me open now', 'hardware stores near me', 'furniture stores in my area',
               'where to buy a mattress near me', 'bike shops near me', 'electronics stores near me',
               'thrift stores near me', 'florists near me with same day delivery', 'pet stores near me',
               'where to buy running shoes near me'],
  'implicit': ['which stores are open on public holidays', 'is there sales tax on clothing',
               'what are my rights to return an online purchase', 'best place to buy a used iPhone',
               'when are the big sales this year', 'best online furniture store', 'which retailers price match',
               'how much does a gallon of milk cost', 'best place to buy a laptop', 'which store has the cheapest groceries'],
  'nonlocal': ['best running shoes', 'best backpack for travel', 'best gifts for teenagers', 'best winter jacket',
               'best office chair', 'best bikes for kids', 'best sunscreen', 'best digital camera for beginners',
               'best yoga mat', 'best luggage brand']},
 'cars and transport': {
  'explicit': ['auto repair shops near me', 'car dealerships near me', 'cheapest gas near me', 'tire shops near me',
               'car wash near me', 'EV charging stations near me', 'driving schools in my area',
               'where to rent a car near me', 'oil change near me', 'used car dealers near me with good reviews'],
  'implicit': ['how much does it cost to register a car', 'best electric car for winter driving',
               'how much is gas right now', 'do I need snow tires', 'how do I renew my driver\'s license',
               'do I need a car to commute', 'best used car under $15,000',
               'how much does car insurance cost for a teen driver', 'what is the speed limit on highways',
               'are there tax credits for buying an electric car'],
  'nonlocal': ['how often should I change my oil', 'best SUV for families', 'how to jump start a car',
               'hybrid or electric car which is better', 'best dash cam', 'how to improve gas mileage',
               'best car seat for infants', 'what does a check engine light mean', 'best all season tires',
               'how long do EV batteries last']},
 'travel and leisure': {
  'explicit': ['things to do near me this weekend', 'hiking trails near me', 'hotels near me', 'gyms near me',
               'yoga studios near me', 'parks near me for kids', 'museums near me', 'movie theaters near me',
               'golf courses near me', 'day trips from my area'],
  'implicit': ['what is the weather like this weekend', 'when is the best time of year to visit the beach',
               'do I need a visa to travel to Europe', 'what are the public holidays this month',
               'best weekend getaway', 'what are the best ski resorts', 'how early should I arrive at the airport',
               'best airline for domestic flights', 'national parks worth visiting', 'how much does a train ticket to the nearest big city cost'],
  'nonlocal': ['how to pack light', 'best travel credit card', 'tips for long flights', 'best travel apps',
               'how to avoid jet lag', 'best carry on suitcase', 'how to find cheap flights',
               'best noise cancelling earbuds for travel', 'what to do if my flight is cancelled', 'best travel pillow']},
 'legal, government and education': {
  'explicit': ['lawyers near me for a will', 'DMV office near me', 'passport office near me', 'daycare centers near me',
               'tutoring services near me', 'best public schools in my area', 'community colleges near me',
               'immigration lawyers near me', 'post office near me', 'divorce lawyers in my area'],
  'implicit': ['how do I get a marriage license', 'what is the legal drinking age', 'how much does college tuition cost',
               'how do I apply for unemployment benefits', 'what are the rules for homeschooling', 'is marijuana legal',
               'how do I get a fishing license', 'how much does it cost to file for divorce',
               'how long does a landlord have to return a security deposit', 'how do I register a small business'],
  'nonlocal': ['how to write a will', 'best online courses for data science', 'how to learn a new language fast',
               'best language learning app', 'how to write a resume', 'best coding bootcamps',
               'how to prepare for a job interview', 'what is a power of attorney', 'best note taking app for students',
               'how to negotiate a salary']},
}


def main():
    rows = []
    for ti, (topic, groups) in enumerate(Q.items()):
        for group, qs in groups.items():
            assert len(qs) == 10, (topic, group, len(qs))
            for i, q in enumerate(qs):
                rows.append(dict(qid=f'{group[:3]}_{ti:02d}_{i:02d}', topic=topic, group=group,
                                 question=q,                    # as a user would type it
                                 pilot=int(i == 0)))
    d = pd.DataFrame(rows)
    assert d.question.is_unique and len(d) == 300
    d.to_csv(DATA / 'questions.csv', index=False)
    print(d.groupby('group').size().to_string(), f'\npilot: {d.pilot.sum()}  -> data/questions.csv')


if __name__ == '__main__':
    main()
