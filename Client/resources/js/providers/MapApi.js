import {BaseApi} from "./BaseApi.js";

export class MapApi extends BaseApi{
    constructor() {
        super(__BASE__PYTHON__URL__);
    }

    async coloringMap(type) {
        try {
            super.httpMethod = 'get';
            super.sourceUrl = '/api/dashboard/regions/sentiment-heatmap?min_reviews=0';
            super.params = {sentiment_filter: type}
            return super.createRequest();
        } catch (e) {
            return e;
        }
    }
}
