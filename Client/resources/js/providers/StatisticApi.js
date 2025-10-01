import {BaseApi} from "./BaseApi.js";

export class StatisticApi extends BaseApi {
    constructor() {
        super(__BASE__PYTHON__URL__);
    }

    async getTableStatistic(dateStart, dateEnd, regionCode = null) {
        try {
            super.httpMethod = 'get';
            super.sourceUrl = '/api/dashboard/regions/products/statistics';
            super.params = {
                date_from: dateStart,
                date_to: dateEnd,
                region_code: regionCode
            }
            return super.createRequest();
        } catch (e) {
            return e;
        }
    }

    async getSourceStatistic(dateStart, dateEnd, regionCode = null) {
        try {
            super.httpMethod = 'get';
            super.sourceUrl = '/api/dashboard/sources/statistics';
            super.params = {
                date_from: dateStart,
                date_to: dateEnd,
                region_code: regionCode
            }
            return super.createRequest();
        } catch (e) {
            return e;
        }
    }

    async getDynamicsOfChanges(dateStart, dateEnd, product, regionCode = null) {
        try {
            super.httpMethod = 'get';
            super.sourceUrl = '/api/dashboard/trends/echarts-data';
            super.params = {
                date_from: dateStart,
                date_to: dateEnd,
                region_code: regionCode,
                product: product
            }
            return super.createRequest();
        } catch (e) {
            return e;
        }
    }

    async getBasicSummary() {
        try {
            super.httpMethod = 'get';
            super.sourceUrl = '/api/dashboard/summary';
            return super.createRequest();
        } catch (e) {
            return e;
        }
    }

    async getDataMatrix() {
        try {
            super.httpMethod = 'get';
            super.sourceUrl = '/api/matrix/product-sentiment-matrix';
            return super.createRequest();
        } catch (e) {
            return e;
        }
    }

    async getCompareData(data) {
        try {
            super.httpMethod = 'post';
            super.sourceUrl = '/api/dashboard/products/sentiment-analysis';
            super.data = data;
            return super.createRequest();
        } catch (e) {
            return e;
        }
    }

    async getGenderProductPrefereces() {
        try {
            super.httpMethod = 'get';
            super.sourceUrl = '/api/dashboard/demographics/gender-product-preferences';
            super.params = {
                analysis_type: 'preferences'
            };
            return super.createRequest();
        } catch (e) {
            return e;
        }
    }
}
