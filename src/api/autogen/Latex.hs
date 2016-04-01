module Latex (generate) where

import Control.Monad (mapM_)
import Data.List (intersperse)
import System.IO
import Text.PrettyPrint

import API
import Common
import qualified Matlab as Matlab
import qualified Cpp as Cpp
import qualified C as C


generate
    :: Bool         -- ^ is this a stand-alone document
    -> [ApiModule]
    -> FilePath     -- ^ output file
    -> IO ()
generate standalone ms file = do
    withFile file WriteMode $ \tex -> do
    if standalone
        then hPutStr tex $ render $ preamble $+$ document (body 0)
        else hPutStr tex $ render $ body 1
    where
        preamble = vcat [cmd1 "documentclass" $ text "article", usepackage "listings"]
        body level = vcat $ map (moduleDoc level) ms



{- Generate documentation for a single module -}
moduleDoc
    :: Int          -- ^ level (sub-section) for the module
    -> ApiModule
    -> Doc
moduleDoc level mdl = section level True (mname ++ " class") (Just mname) $
        vcat $ [description, ctor_doc, fn_list, pagebreak] ++ fns_doc
    where
        mname = name mdl
        description = maybe empty text $ describe mdl
        ctor_doc = moduleConstructorDoc mdl
        fn_list = paragraph "Functions" $ moduleFunctionList fns
        fns_doc = map (functionDoc mdl (level+1)) $ fns

        {- Possibly, not all functions should be documented -}
        fns = filter (not . elem LaTeX . fn_noauto) $ mdl_functions mdl



{- Print a code block showing how to construct this for each API language -}
moduleConstructorDoc :: ApiModule -> Doc
moduleConstructorDoc mdl = vcat $ (nl:) $ intersperse nl $ map go [CPP, C, Matlab]
    where
        go :: Language -> Doc
        go Matlab = codeblock Matlab $ Matlab.ctorSynopsis mdl
        go CPP = codeblock CPP $ Cpp.ctorSynopsis mdl
        go C = codeblock C $ C.ctorSynopsis mdl



{- Print a list of all the functions in a module -}
moduleFunctionList :: [ApiFunction] -> Doc
moduleFunctionList = itemize . map go
    where
        go :: ApiFunction -> Doc
        go fn = hyperref (functionLabel fn) (name fn)



functionDoc :: ApiModule -> Int -> ApiFunction -> Doc
functionDoc mdl level fn = section level False fname lbl $
        vcat [brief, nl, synopsis, inputs, outputs, description, nl, pagebreak]
    where
        lbl = Just $ functionLabel fn
        fname = name mdl ++ "::" ++ name fn
        brief = text $ fn_brief fn
        synopsis = synopsisDoc mdl (level+1) fn
        inputs = inputDoc (level+1) $ fn_inputs fn
        outputs = outputDoc (level+1) $ fn_output fn
        description = maybe empty text $ describe fn


functionLabel :: ApiFunction -> String
functionLabel fn = "fn: " ++ name fn


-- TODO: use definition list instead?
inputDoc :: Int -> [Input] -> Doc
inputDoc level [] = empty
inputDoc level xs = section level False "Inputs" Nothing $ description (map (go . arg) xs)
    where
        -- TODO: deal with optional arguments here
        go :: ApiArg -> (Doc, Doc)
        go arg = (text $ texFormat $ name arg, maybe empty text $ describe arg)


-- Perform replacements of TeX-specific symbols
-- TODO: use a more general search-and-replace here
texFormat :: String -> String
texFormat [] = []
texFormat [x] = [x]
texFormat (x1:x2:xs)
    | x1 /= '\\' && x2 == '_'= x1 : texFormat ('\\' : x2 : xs)
    | otherwise              = x1 : texFormat (x2 : xs)



outputDoc :: Int -> [OutputType] -> Doc 
outputDoc level [] = empty
outputDoc level xs = section level False "Outputs" Nothing $ description (map go xs)
    where
        -- TODO: share this code with inputDoc
        go :: ApiArg -> (Doc, Doc)
        go arg = (text $ name arg, maybe empty text $ describe arg)


synopsisDoc :: ApiModule -> Int -> ApiFunction -> Doc
synopsisDoc mdl level fn = vcat $ (nl:) $ intersperse nl $ map go [CPP, C, Matlab]
    where
        go :: Language -> Doc
        go Matlab = codeblock Matlab $ Matlab.synopsis mdl fn
        go CPP = codeblock CPP $ Cpp.synopsis mdl fn
        go C = codeblock C $ C.synopsis mdl fn



nl = text ""

cmd :: String -> Doc
cmd name = text "\\" <> text name


cmd1 :: String -> Doc -> Doc
cmd1 name arg = (cmd name) <> braces arg

-- cmda :: String -> [Doc] -> Doc
-- cmda name args = cmd name <> (braces $ hcat $ punctuate (char ',') args)


section :: Int -> Bool -> String -> Maybe String -> Doc -> Doc
section level number title labelString body
    | level > 2 = paragraph title body
    | otherwise = cmd1 cmdname (text title) $+$ lbl $+$ body
    where
        cmdname = subs ++ "section" ++ star
        subs = concat $ replicate level "sub"
        star = if number then "" else "*"
        lbl = maybe empty label labelString


paragraph :: String -> Doc -> Doc
paragraph title body = cmd1 "paragraph" (text title) $+$ body



document = env "document" []


label :: String -> Doc
label = cmd1 "label" . text


begin :: String -> [String] -> Doc
begin name [] = cmd1 "begin" $ text name
begin name optargs = (begin name []) <> (arglistWith brackets $ map text optargs)


end :: String -> Doc
end = cmd1 "end" . text


-- TODO: add arguments here
env :: String -> [String] -> Doc -> Doc
env name args body = vcat [begin name args, body, end name]


itemize :: [Doc] -> Doc
itemize [] = empty
itemize xs = env "itemize" [] $ vcat $ map item xs

item :: Doc -> Doc
item x = cmd "item" <+> x


description :: [(Doc, Doc)] -> Doc
description [] = empty
description xs = env "description" [] $ vcat $ map descr_item xs

descr_item :: (Doc, Doc) -> Doc
descr_item (name, descr) = cmd "item" <> brackets name <+> descr


codeline :: Doc -> Doc
codeline = cmd1 "texttt"


codeblock :: Language -> Doc -> Doc
codeblock lang code = vcat [title, listing]
    where
        -- title = cmd1 "flushleft" $ text $ (name lang) ++ ":"
        title = cmd "noindent" <+> (text $ (name lang) ++ ":")
        listing = env "lstlisting" ["aboveskip=2pt"] code



pagebreak = cmd "clearpage"


usepackage :: String -> Doc
usepackage = cmd1 "usepackage" . text


flushleft :: Doc -> Doc
flushleft = env "flushleft" []


hyperref :: String -> String -> Doc
hyperref label str = cmd "hyperref" <> brackets (text label) <> braces (text str)
