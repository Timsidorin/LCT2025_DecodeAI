from fast_depends import Depends
import ReviewService
from ReviewParser import ReviewMonitor as RM
from main import get_review_service
from shemas.review import ReviewCreate
import asyncio

parser = RM()


async def get_and_publish_reviews():
    review_service = await get_review_service()

    result = parser.get_new_reviews()
    print(result)
    if result:
        for review in result['reviews']:
            serialized_review = ReviewCreate(
                source_id=review['source_id'],
                text=review['text'],
                datetime_review=review['datetime_review'],
                rating=None,
                product=None
            )
            await review_service.create_review(serialized_review)



