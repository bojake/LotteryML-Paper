import os
import datetime as dt
from dataclasses import dataclass

@dataclass
class LottoConfigEntry:
    key: str
    game: str
    start_date: dt.datetime
    end_date: dt.datetime
    normalizer: int
    maxval: list[int]
    fname: str
    title: str
    cols: list[str]
    draw_days: list[int]
    odds: int
    notable_dates: list[str]

class LottoConfig(object) :
    def __init__(self) :
        # monday is 0
        self.lottocfg = {
            "powerball" : LottoConfigEntry(**{
                "key" : "powerball",
                "game":"powerball",
                "start_date" : dt.datetime(year=2015,month=10,day=4),
                "end_date" : None,
                "normalizer" : 69,
                "maxval" : [69,69,69,69,69,26],
                "fname": "powerball_numbers_current.csv",
                "title" : "Powerball 2000-2022 Draw Bias",
                "cols" :['D1','D2','D3','D4','D5','PowerBall'],
                "draw_days" : [0, 2, 5], 
                "odds" : 292201338,
                "notable_dates" : ['2023-10-11','2023-07-19','2019-03-27','2016-01-13']
            }),
            "powerballpre2008" : LottoConfigEntry(**{
                "key" : "powerballpre2008",
                "game":"powerball",
                "start_date" : dt.datetime(year=1997,month=11,day=1),
                "end_date" : dt.datetime(year=2008,month=11,day=12),
                "normalizer" : 55,
                "maxval" : [55,55,55,55,55,42],
                "fname": "powerball_numbers_current.csv",
                "title" : "Powerball 2000-2022 Draw Bias",
                "cols" :['D1','D2','D3','D4','D5','PowerBall'],
                "draw_days" : [0, 2, 5], 
                "odds" : 292201338,
                "notable_dates" : ['2005-03-30','2007-08-25','2006-02-18','2005-10-19','2002-12-25']
            }),
            "powerballpre2012" : LottoConfigEntry(**{
                "key" : "powerballpre2012",
                "game":"powerball",
                "start_date" : dt.datetime(year=2008,month=11,day=12),
                "end_date" : dt.datetime(year=2012,month=1,day=15),
                "normalizer" : 59,
                "maxval" : [59,59,59,59,59,39],
                "fname": "powerball_numbers_current.csv",
                "title" : "Powerball 2000-2022 Draw Bias",
                "cols" :['D1','D2','D3','D4','D5','PowerBall'],
                "draw_days" : [0, 2, 5], 
                "odds" : 292201338,
                "notable_dates" : ['2011-11-19']
            }),
            "powerball2012to2015" : LottoConfigEntry(**{
                "key" : "powerball2012to2015",
                "game":"powerball",
                "start_date" : dt.datetime(year=2012,month=1,day=15),
                "end_date" : dt.datetime(year=2015,month=10,day=4),
                "normalizer" : 59,
                "maxval" : [59,59,59,59,59,35],
                "fname": "powerball_numbers_current.csv",
                "title" : "Powerball 2000-2022 Draw Bias",
                "cols" :['D1','D2','D3','D4','D5','PowerBall'],
                "draw_days" : [0, 2, 5], 
                "odds" : 292201338,
                "notable_dates" : ['2013-05-18']
            }),
            "mega" : LottoConfigEntry(**{
                "key" : "mega",
                "end_date" : None,
                "start_date" : None,
                "game":"mega",
                "normalizer" : 70,
                "maxval": [70,70,70,70,70,25],
                "fname": "mega_numbers_current.csv",
                "title" : "Mega 2005-2022 Draw Bias",
                "cols" :['D1','D2','D3','D4','D5','Mega'],
                "draw_days" : [1,4],
                "odds": 0,
                "notable_dates" : []
            }),
            "superlotto" : LottoConfigEntry(**{
                "key" : "superlotto",
                "game":"superlotto",
                "end_date" : None,
                "start_date" : None,
                "normalizer" : 47,
                "maxval": [47,47,47,47,47,27],
                "fname": "superlotto_numbers_current.csv",
                "title" : "SuperLotto 2000-2022 Draw Bias",
                "cols" :['D1','D2','D3','D4','D5','Mega'],
                "draw_days" : [2,5],
                "odds" : 0,
                "notable_dates" : []
            }),
            "fantasy5" : LottoConfigEntry(**{
                "key" : "fantasy5",
                "game":"fantasy5",
                "end_date" : None,
                "start_date" : None,
                "normalizer" : 39,
                "maxval": [39,39,39,39,39],
                "fname": "fantasy5_numbers_current.csv",
                "title" : "Fantasy 5 2000-2022 Draw Bias",
                "cols" :['D1','D2','D3','D4','D5'],
                "draw_days" : [0,1,2,3,4,5,6],
                "odds" : 0,
                "notable_dates" : []
            })
        }
    
    def keys(self) :
        return self.lottocfg.keys()
    
    def dow(self,lotto) :
        return self.lottocfg[lotto]['draw_days']
    
    def lotto_splits(self, lotto:str) -> list[LottoConfigEntry] :
        splits = []
        for xx in self.lottocfg.keys() :
            if self.lottocfg[xx].game == lotto :
                splits.append(self.lottocfg[xx])
        return splits
        
    def __getitem__(self,tuple_index) -> LottoConfigEntry :
        index = None
        dte = None
        if len(tuple_index) == 2:
            index,dte = tuple_index
        else :
            index = tuple_index
        if not dte is None :
            for xx in self.lottocfg.keys() :
                if self.lottocfg[xx].game == index :
                    if self.lottocfg[xx].start_date is not None and dte >= self.lottocfg[xx].start_date:
                        if self.lottocfg[xx].end_date is not None :
                            if dte < self.lottocfg[xx].end_date : # self.lottocfg[xx]['end-date'] :
                                return self.lottocfg[xx]
                        else:
                            return self.lottocfg[xx]
        return self.lottocfg[index.lower()]
    
    def powerball(self) -> LottoConfigEntry:
        return self.lottocfg['powerball']

    def mega(self) -> LottoConfigEntry:
        return self.lottocfg['mega']

    def superlotto(self) -> LottoConfigEntry:
        return self.lottocfg['superlotto']

    def fantasy5(self) -> LottoConfigEntry:
        return self.lottocfg['fantasy5']

