"""Fixed implementation of the financial news retrieval methods for the ComprehensiveAnalysisOrchestrator."""

def _retrieve_financial_news(self,
                           symbol: str,
                           options: dict,
                           result):
    """Retrieve financial news for a symbol.
    
    Args:
        symbol: Stock symbol
        options: Analysis options
        result: Result object to populate
        
    Raises:
        DataRetrievalError: If data cannot be retrieved
    """
    from stock_analysis.utils.exceptions import DataRetrievalError
    
    news_limit = options.get('news_limit', 10)
    news_days = options.get('news_days', 7)
    include_sentiment = options.get('sentiment', True)
    
    self.logger.info(f"Retrieving financial news for {symbol} (limit: {news_limit}, days: {news_days}, sentiment: {include_sentiment})")
    
    try:
        # Get company news
        self.logger.debug(f"Retrieving company news for {symbol}")
        news_items = self.news_service.get_company_news(
            symbol=symbol,
            days=news_days,
            limit=news_limit,
            include_sentiment=include_sentiment,
            use_cache=True
        )
        
        if news_items:
            result.news_items = news_items
            self.logger.debug(f"Successfully retrieved {len(news_items)} news items for {symbol}")
        else:
            self.logger.warning(f"No news items found for {symbol}")
            result.news_items = []
        
        # Get news sentiment if requested
        if include_sentiment:
            try:
                self.logger.debug(f"Retrieving news sentiment for {symbol}")
                news_sentiment = self.news_service.get_news_sentiment(
                    symbol=symbol,
                    days=news_days,
                    use_cache=True
                )
                
                if news_sentiment:
                    result.news_sentiment = news_sentiment
                    self.logger.debug(f"Successfully retrieved news sentiment for {symbol}: {news_sentiment.overall_sentiment:.2f}")
                else:
                    self.logger.warning(f"No sentiment data available for {symbol}")
                    result.news_sentiment = None
            except Exception as e:
                self.logger.error(f"Failed to retrieve news sentiment for {symbol}: {str(e)}")
                self.logger.debug(f"News sentiment error details: {type(e).__name__}", exc_info=True)
                if not self.continue_on_error:
                    raise DataRetrievalError(
                        f"Failed to retrieve news sentiment for {symbol}: {str(e)}",
                        symbol=symbol,
                        data_source="news_sentiment"
                    )
    except Exception as e:
        self.logger.error(f"Failed to retrieve financial news for {symbol}: {str(e)}")
        self.logger.debug(f"Financial news error details: {type(e).__name__}", exc_info=True)
        if not self.continue_on_error:
            raise DataRetrievalError(
                f"Failed to retrieve financial news for {symbol}: {str(e)}",
                symbol=symbol,
                data_source="financial_news"
            )

def _retrieve_financial_news_task(self,
                                symbol: str,
                                options: dict):
    """Retrieve financial news for a symbol as a parallel task.
    
    Args:
        symbol: Stock symbol
        options: Analysis options
        
    Returns:
        Dictionary containing news items and sentiment
        
    Raises:
        DataRetrievalError: If data cannot be retrieved
    """
    from datetime import datetime
    from stock_analysis.utils.exceptions import DataRetrievalError
    
    news_limit = options.get('news_limit', 10)
    news_days = options.get('news_days', 7)
    include_sentiment = options.get('sentiment', True)
    
    self.logger.debug(f"Financial news task for {symbol} (limit: {news_limit}, days: {news_days}, sentiment: {include_sentiment})")
    
    result = {
        'news_items': [],
        'news_sentiment': None,
        '_metadata': {
            'symbol': symbol,
            'news_limit': news_limit,
            'news_days': news_days,
            'include_sentiment': include_sentiment,
            'timestamp': datetime.now()
        }
    }
    
    try:
        # Get company news
        news_items = self.news_service.get_company_news(
            symbol=symbol,
            days=news_days,
            limit=news_limit,
            include_sentiment=include_sentiment,
            use_cache=True
        )
        
        if news_items:
            result['news_items'] = news_items
            self.logger.debug(f"Retrieved {len(news_items)} news items for {symbol}")
        else:
            self.logger.warning(f"No news items found for {symbol}")
        
        # Get news sentiment if requested
        if include_sentiment:
            try:
                news_sentiment = self.news_service.get_news_sentiment(
                    symbol=symbol,
                    days=news_days,
                    use_cache=True
                )
                
                if news_sentiment:
                    result['news_sentiment'] = news_sentiment
                    self.logger.debug(f"Retrieved news sentiment for {symbol}: {news_sentiment.overall_sentiment:.2f}")
                else:
                    self.logger.warning(f"No sentiment data available for {symbol}")
            except Exception as e:
                self.logger.error(f"Failed to retrieve news sentiment for {symbol}: {str(e)}")
                self.logger.debug(f"News sentiment error details: {type(e).__name__}", exc_info=True)
                if not self.continue_on_error:
                    raise DataRetrievalError(
                        f"Failed to retrieve news sentiment for {symbol}: {str(e)}",
                        symbol=symbol,
                        data_source="news_sentiment"
                    )
    except Exception as e:
        self.logger.error(f"Failed to retrieve financial news for {symbol}: {str(e)}")
        self.logger.debug(f"Financial news error details: {type(e).__name__}", exc_info=True)
        if not self.continue_on_error:
            raise DataRetrievalError(
                f"Failed to retrieve financial news for {symbol}: {str(e)}",
                symbol=symbol,
                data_source="financial_news"
            )
    
    return result