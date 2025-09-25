import {BaseApi} from "./BaseApi.js";

export class MapApi extends BaseApi{
    constructor() {
        super(__BASE__PYTHON__URL__);
    }

    async coloringMap() {
        try {
            super.httpMethod = 'get';
            super.sourceUrl = '/api/dashboard/regions/sentiment-heatmap?sentiment_filter=neutral&min_reviews=0';
            return super.createRequest();
        } catch (e) {
            return e;
        }
    }
}
