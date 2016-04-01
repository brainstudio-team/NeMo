module Main where

import API (network, simulation, configuration, matlabExtras, constructable)
import qualified Matlab (generate)
import qualified Latex (generate)
-- import qualified Haskell (generate)
import qualified Python (generate)

main = do
    -- Latex.generate True api "tmp/fnref.tex"
    -- Latex.generate False api "../../doc/latex/fnref.tex"
    Matlab.generate (api ++ [matlabExtras]) constructable
    -- Haskell.generate api
    Python.generate api constructable
    where
        api = [network, configuration, simulation]
