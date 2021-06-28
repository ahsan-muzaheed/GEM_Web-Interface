const { response } = require("express");


// Load basic GET pages
exports.getLandingPage = (req, res, next) => {
    
    console.log('Landing called!');
    
    
    res.render('landing', {
        pageTitle: 'Landing'
    });
    
   
    
};

exports.getLoadingPage = (req, res, next) => {
    console.log('loading page called!');
    res.render('loading', {
        pageTitle: 'Loading'
    });
}

exports.getGraphEmbeddingPage = (req, res, next) => {
    
    res.render('graphEmbedding', {
        pageTitle: 'GraphEmbedding'
    });
}

exports.getDownStreamMLPage = (req, res, next) => {
    res.render('downStream', {
        pageTitle: 'downStream'
    });
}
exports.getResultPage = (req, res, next) => {
    res.render('result', {
        pageTitle: 'result'
    });
}