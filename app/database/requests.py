#This file is useless since the AI is free and there are no any instructions how to connect this file to postgresql

from app.database.models import async_session
from app.database.models import User, AiModel
from sqlalchemy import select, update, delete, desc
from decimal import Decimal

def connection(func):
    async def inner(*args, **kwargs):
        async with async_session() as session:
            return await func(session, *args, **kwargs)
    return inner

async def set_user(tg_id, session):
    user = await session.scalar(select(User).where(User.tg_id == tg_id))
        
    if not user:
        session.add(User(tg_id=tg_id, balance='0'))
        await session.commit()

         
async def get_user(tg_id, session):
    return await session.scalar(select(User).where(User.tg_id == tg_id))

async def get_users(session):
    return await session.scalars(select(User))
    
async def calculate(tg_id, session, sum, model_name):
    user = await session.scalar(select(User).where(User.tg_id == tg_id))
    model = await session.scalar(select(AiModel).where(AiModel.name == model_name))
    new_balance = Decimal(Decimal(user.balance) - (Decimal(model.price) * Decimal(sum)))
    await session.execute(update(User).where(User.id == user.id).values(balance=str(new_balance)))
    await session.commit()