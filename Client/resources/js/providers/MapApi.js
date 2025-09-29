import {BaseApi} from "./BaseApi.js";

export class MapApi extends BaseApi{
    constructor() {
        super(__BASE__PYTHON__URL__);
    }

    async coloringMap(type) {
        try {
            super.httpMethod = 'get';
            super.sourceUrl = '/api/dashboard/sentiment/heatmap';
            super.params = {sentiment_filter: type}
            return super.createRequest();
        } catch (e) {
            return e;
        }
    }
}
