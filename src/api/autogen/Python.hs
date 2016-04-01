module Python (generate) where

import Text.PrettyPrint
import System.IO
import Data.Char
import Data.Maybe (maybe)
import Data.List (intercalate)
import Network.URI (escapeURIString)

import API
import Common

-- TODO: generate CPP macros instead

{- The API is generated using boost::python. All we need to do here is generate
 - the docstring -}
-- TODO: could generate most methods here, just except overloaded methods.
generate
    :: [ApiModule]
    -> [ApiFunction] -- ^ functions common to both Simulation and Network
    -> IO ()
generate ms constructable =
    withFile "../python/docstrings.h" WriteMode $ \hdl -> do
    hPutStr hdl $ render $ vcat (map moduleDoc ms) <> text "\n"
    hPutStr hdl $ render $ vcat (map (functionDoc "CONSTRUCTABLE") constructable) <> text "\n"


docstringStyle = Style PageMode 75 1.0

{- | Turn a string into a block of text with a fixed width and with escaped
 - newlines -}
docRender :: String -> Doc
docRender = text . intercalate "\\n" . lines . renderStyle docstringStyle . fsep . map text . words


{- Generate global static named docstrings for each method -}
moduleDoc :: ApiModule -> Doc
moduleDoc mdl = classDoc $+$ functionDocs
    where
        classDoc = text "#define" <+> macroName <+> classDocBody
        macroName = underscoredUpper [mdl_name mdl, "doc"]
        classDocBody = doubleQuotes $ maybe empty docRender (mdl_descr mdl)
        functionDocs = vcat $ map (functionDoc (name mdl)) $ mdl_functions mdl


{- TODO: perhaps use actual formatting characters here -}
functionDoc :: String -> ApiFunction -> Doc
functionDoc mname fn = text "#define" <+> macroName <+> docstring
    where
        macroName = underscoredUpper [mname, camelToUnderscore (name fn), "doc"]
        docstring = doubleQuotes $ mergeLinesWith "\\n\\n" $ empty : filter (not . isEmpty) [synopsis, inputs, ret, description]
        synopsis = text $ fn_brief fn
        inputs = inputDoc $ fn_inputs fn
        ret = outputDoc $ fn_output fn
        description = maybe empty (docRender . escape) $ describeLanguage Python fn


inputDoc :: [Input] -> Doc
inputDoc [] = empty
inputDoc xs = mergeLines $ (text "Inputs:" : map (go . arg) xs)
    where
        -- TODO: deal with optional arguments here
        go :: ApiArg -> Doc
        go arg = text (name arg) <+> maybe empty (\a -> text "--" <+> text a) (describe arg)


outputDoc :: [ApiArg] -> Doc
outputDoc [] = empty
outputDoc [x] = text "Returns" <+> maybe empty text (describe x)
outputDoc _ = error "Documentation for multiple output arguments not supported"


-- functionName :: ApiFunction -> Doc
-- functionName = text . camelToUnderscore . name


-- qualifiedFunctionName :: String -> ApiFunction -> Doc
-- qualifiedFunctionName moduleName fn = text moduleName <> char '_' <> functionName fn


escape :: String -> String
escape xs = go xs
    where
        go [] = []
        go ('"':xs) = '\\' : '"' : go xs
        go (x:xs) = x : go xs
